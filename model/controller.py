import math
import os
import pandas as pd

from model.model_zoo.resnet_local import resnet50

class Code2SpacePartition:
    def __init__(self, devcie_name="cpu", model_name="resnet_50"):
        """Load model profiles"""
        model_info = pd.read_excel(os.path.dirname(os.path.realpath(__file__)) + "/model_zoo/model_info.xlsx", sheet_name=model_name)
        self.layer_name = model_info['layer_name'].values
        self.layer_size = model_info['layer_size'].values
        self.pi_latency = model_info['pi_latency'].values
        self.server_latency = model_info['server_' + devcie_name + '_latency'].values
        self.BW_RANGE = [50, 600] # (Mbps) used for normalization
        self.TRANSMISSION_DELAY = [0.020, 4] # network delay when using repeater satellites

    def init_model_server(self, partition_idx):
        """initialize the model instance executed on the local server"""
        if partition_idx is None or (partition_idx == len(self.layer_name)-1):
            # switch to satellite-only execution
            return None, self.layer_name[-1]
        else:
            # create the model instance
            net = resnet50(first_layer=self.layer_name[partition_idx], end_layer='prediction')
        return net, self.layer_name[partition_idx]

    def init_model_pi(self, partition_layer_name):
        """initialize the model instance executed on the satellite"""
        if partition_layer_name == self.layer_name[0]:
            # switch to local-only execution
            return None
        else:
            net = resnet50(first_layer='input', end_layer=partition_layer_name)
        return net

    def get_abs_max_min_e2e(self, server_ratio, cpu_freq_ratio, trans_delay_range):
        """
        Depend on the available resources of the local server and the satellite, and transmission delay (optional)
        ,incurred by repeater satellites,to evaluate the maximum and the minimum end-to-end latency
        :param server_ratio: available resources of the local server
        :param cpu_freq_ratio: available resources of the satellite
        :param trans_delay_range: maximum/minimum network delay when using repeater satellites to communicate data
        :return: minimum and maximum end-to-end latency
        """
        abs_min_e2e, abs_max_e2e = None, None
        # evaluate the end-to-end latency under given partition layers
        for partition_idx, partition_layer in enumerate(self.layer_name):
            if partition_idx != 8:
                # network latency, including transmission delay, propagation delay,
                # and network delay (optional) incurred by repeater satellites based communication
                min_net_latency = self.layer_size[partition_idx] / (
                        self.BW_RANGE[1] / 8.0) + 0.007 + trans_delay_range[0]
                max_net_latency = self.layer_size[partition_idx] / (
                        self.BW_RANGE[0] / 8.0) + 0.007 + trans_delay_range[1]
            else:
                # no network latency under satellite-only execution
                min_net_latency = 0
                max_net_latency = 0
            # inference latency of the model layers running on the satellite and the local server
            pi_latency, server_latency = self.pi_latency[partition_idx] / cpu_freq_ratio, \
                                         self.server_latency[partition_idx] * server_ratio

            min_e2e = pi_latency + min_net_latency + server_latency
            max_e2e = pi_latency + max_net_latency + server_latency

            if partition_idx == 0:
                abs_min_e2e, abs_max_e2e = min_e2e, max_e2e
            else:
                if min_e2e < abs_min_e2e:
                    abs_min_e2e = min_e2e
                if max_e2e > abs_max_e2e:
                    abs_max_e2e = max_e2e
        return abs_min_e2e, abs_max_e2e

    def get_max_min_band(self, network_bw, cpu_freq_ratio):
        """
        Evaluate the maximum/minimum network occupation (used for normalization)
        :param network_bw: network bandwidth
        :param cpu_freq_ratio: available CPU resources of the satellite
        :return:
        """
        min_band, max_band = None, None
        for partition_idx, partition_layer in enumerate(self.layer_name):
            pi_latency = self.pi_latency[partition_idx] / cpu_freq_ratio
            band_occupation = self.layer_size[partition_idx] / (network_bw / 8.0) * 1 / min(33.33, pi_latency)
            if partition_idx == 0:
                min_band, max_band = band_occupation, band_occupation
            else:
                if band_occupation < min_band:
                    min_band = band_occupation
                if band_occupation > max_band:
                    max_band = band_occupation
        return min_band, max_band

    def get_partition_point(self, network_bw, overall_thr_weight, mobile_thr_weight,
                            cpu_freq_ratio, server_ratio, transmission=0):
        """
        Depend on the network bandwidth, and the weights for overall thr., mobile thr., and network occupation, to get
        the partition layer that can maximize the utility of the system.
        Here, utility is alpha/(L_m+L_n+L_e)+ beta/L_m + gamma/L_n where alpha+beta+gamma = 1

        :param network_bw: (Mbps) the down link bandwidth when returning data from the satellite to the ground
        :param cpu_freq_ratio: available resources of the satellite, calculated as current CPU freq / maximum CPU freq
        :param server_ratio: available resources of the local server
        :param overall_thr_weight: weight for overall throughput, where overall thr. equals to 1/(L_m+L_n+L_e).
        Note that L_m, L_n, and L_e stand for mobile latency, network latency and edge latency, respectively.
        :param mobile_thr_weight: weight for mobile throughput computed as 1/L_m
        :param transmission: specific network latency when using repeater satellites to communicate data
        :return: max_idx=> partition layer idx, max_utility.
        Note that return None if model is executed as satellite-only, when only prediction results are uploaded to the ground.
        Otherwise, return the detailed partition point
        """
        # 1. check the input parameters
        if overall_thr_weight < 0 or mobile_thr_weight < 0 or (overall_thr_weight + mobile_thr_weight) > 1:
            print("invalid weights", overall_thr_weight, mobile_thr_weight)
            return None, None
        if transmission != 0 and (
                transmission < self.TRANSMISSION_DELAY[0] or transmission > self.TRANSMISSION_DELAY[1]):
            print("invalid transmission delay", transmission)
            return None, None
        # 2. evaluate the maximum and minimum end-to-end latency and bandwidth occupation for the following normalization
        max_utility, max_idx = None, 0
        if transmission == 0:
            min_e2e, max_e2e = self.get_abs_max_min_e2e(server_ratio, cpu_freq_ratio, [0, 0])
        else:
            min_e2e, max_e2e = self.get_abs_max_min_e2e(server_ratio, cpu_freq_ratio, self.TRANSMISSION_DELAY)
        min_band, max_band = self.get_max_min_band(network_bw, cpu_freq_ratio)

        # 3. obtain the achieved utility under given partition layers
        for partition_idx, partition_layer in enumerate(self.layer_name):
            utility = self._get_utility(network_bw, partition_idx, overall_thr_weight, mobile_thr_weight,
                                                cpu_freq_ratio, server_ratio, min_e2e, max_e2e, min_band, max_band,
                                                transmission)
            if max_utility is None or utility > max_utility:
                max_utility = utility
                max_idx = partition_idx
        return max_idx, max_utility

    def _get_utility_details(self, network_bw, partition_idx, overall_thr_weight, mobile_thr_weight,
                             cpu_freq_ratio, server_ratio, min_e2e, max_e2e, min_band, max_band, transmission):
        """
        Please refer to the document to check the detailed formulation.
        :param network_bw: (Mbps) the down link bandwidth when returning data from the satellite to the ground
        :param partition_idx: the index of the partition layer
        :param overall_thr_weight: weight for overall throughput, where overall thr. equals to 1/(L_m+L_n+L_e).
        Note that L_m, L_n, and L_e stand for mobile latency, network latency and edge latency, respectively.
        :param mobile_thr_weight: weight for mobile throughput computed as 1/L_m
        :param cpu_freq_ratio: available resources of the satellite, calculated as current CPU freq / maximum CPU freq
        :param server_ratio: available resources of the local server
        :param min_e2e: the minimum end-to-end latency
        :param max_e2e: the maximum end-to-end latency
        :param min_band: the minimum network occupation
        :param max_band: the maximum network occupation
        :param transmission: specific network latency when using repeater satellites to communicate data
        :return: the detailed utility
        """
        pi_latency, server_latency = self.pi_latency[partition_idx] / cpu_freq_ratio, \
                                     self.server_latency[partition_idx] * server_ratio
        band_occupation = self.layer_size[partition_idx] / (network_bw / 8.0) * 1 / min(33.33, pi_latency)
        if partition_idx == 8:
            net_latency = 0
        else:
            net_latency = self.layer_size[partition_idx] / (network_bw / 8.0) + 0.007 + transmission
        # normalize the achieved end-to-end latency
        norm_e2e = (math.tanh(1 / (pi_latency + net_latency + server_latency)) - math.tanh(1 / max_e2e)) / (
                math.tanh(1 / min_e2e) - math.tanh(1 / max_e2e))
        # normalize the throughput of the satellite
        norm_pi = (math.tanh(1 / pi_latency) - math.tanh(1 / self.pi_latency[-1])) / (
                math.tanh(1 / self.pi_latency[0]) - math.tanh(1 / self.pi_latency[-1]))
        #  normalize the bandwidth occupation
        norm_network = (math.tanh(1 / band_occupation) - math.tanh(1 / max_band)) / (
                math.tanh(1 / min_band) - math.tanh(1 / max_band))
        return [overall_thr_weight * norm_e2e, mobile_thr_weight * norm_pi,
                (1 - overall_thr_weight - mobile_thr_weight) * norm_network]

    def _get_utility(self, network_bw, partition_idx, overall_thr_weight, mobile_thr_weight,
                     cpu_freq_ratio, server_ratio, min_e2e, max_e2e, min_band, max_band, transmission):

        details = self._get_utility_details(network_bw, partition_idx, overall_thr_weight, mobile_thr_weight,
                                            cpu_freq_ratio, server_ratio, min_e2e, max_e2e, min_band, max_band, transmission)

        return sum(details)
