import math
import os
import pandas as pd

from model.model_zoo.resnet_local import resnet50

class Code2SpacePartition:
    def __init__(self, devcie_name="cpu", model_name="resnet_50"):
        model_info = pd.read_excel(os.path.dirname(os.path.realpath(__file__)) + "/model_zoo/model_info.xlsx", sheet_name=model_name)
        self.layer_name = model_info['layer_name'].values
        self.layer_size = model_info['layer_size'].values  # output size in MB
        self.pi_latency = model_info['pi_latency'].values
        self.server_latency = model_info['server_' + devcie_name + '_latency'].values

    def init_model_server(self, network_bw, overall_thr_weight=0.4, mobile_thr_weight=0.2, CPU_freq=1, server_ratio=1):
        # 1. get partition layer
        partition_idx, utility = self.get_partition_point(network_bw, overall_thr_weight, mobile_thr_weight, CPU_freq, server_ratio)

        # 2. depend on partition idx to initialize the model instance
        if partition_idx == len(self.layer_name):
            return None
        else:
            net = resnet50(first_layer=self.layer_name[partition_idx], end_layer='prediction')
        return net, self.layer_name[partition_idx]

    def init_model_pi(self, partition_layer_name):
        if partition_layer_name == self.layer_name[0]:
            return None
        else:
            net = resnet50(first_layer='input', end_layer=partition_layer_name)
        return net, partition_layer_name

    def get_max_min_e2e(self, network_bw, server_ratio, latency_ratio=1):
        min_e2e, max_e2e = None, None
        for partition_idx, partition_layer in enumerate(self.layer_name):
            net_latency = self.layer_size[partition_idx] / (
                        network_bw / 8.0) + 0.007  # layer size in MB, network bandwidth as Mbps
            pi_latency, server_latency = self.pi_latency[partition_idx] * latency_ratio, \
                                         (self.server_latency[-1] - self.server_latency[
                                             partition_idx]) * latency_ratio * server_ratio
            e2e = pi_latency + net_latency + server_latency
            if partition_idx == 0:
                min_e2e, max_e2e = e2e, e2e
            else:
                if e2e < min_e2e:
                    min_e2e = e2e
                if e2e > max_e2e:
                    max_e2e = e2e
        return min_e2e, max_e2e

    def get_max_min_band(self, network_bw, latency_ratio=1):
        min_band, max_band = None, None
        for partition_idx, partition_layer in enumerate(self.layer_name):
            pi_latency = self.pi_latency[partition_idx] * latency_ratio
            band_occupation = self.layer_size[partition_idx] / (network_bw / 8.0) * 1 / min(33.33, pi_latency)
            if partition_idx == 0:
                min_band, max_band = band_occupation, band_occupation
            else:
                if band_occupation < min_band:
                    min_band = band_occupation
                if band_occupation > max_band:
                    max_band = band_occupation
        return min_band, max_band

    def get_partition_point(self, network_bw, overall_thr_weight, mobile_thr_weight, CPU_freq, server_ratio):
        """
        Depend on the network bandwidth, and the weights for overall thr., mobile thr., and network thr, to get
        the partition layer that can maximize the utility of the system.
        Here, utility is alpha/(L_m+L_n+L_e)+ beta/L_m + gamma/L_n where alpha+beta+gamma = 1

        :param network_bw: (Mbps) the down link bandwidth when returning data from the space to the ground
        :param overall_thr_weight: weight for overall throughput, where overall thr. equals to 1/(L_m+L_n+L_e).
        Note that L_m, L_n, and L_e stand for mobile latency, network latency and edge latency, respectively.
        :param mobile_thr_weight: weight for mobile throughput computed as 1/L_m
        :return: max_idx=> partition layer idx, max_utility.
        Note that return None if model is executed as mobile-only when only prediction results are uploaded to the ground.
        Otherwise, return the detailed partition point
        """
        # 1. check the input parameters
        if overall_thr_weight < 0 or mobile_thr_weight < 0 or (overall_thr_weight + mobile_thr_weight) > 1:
            print("invalid weights", overall_thr_weight, mobile_thr_weight)
            return
        # 2. iterate the candidate partition layer
        max_utility, max_idx = None, 0
        min_e2e, max_e2e = self.get_max_min_e2e(network_bw, server_ratio)
        min_band, max_band = self.get_max_min_band(network_bw)
        for partition_idx, partition_layer in enumerate(self.layer_name):
            utility = self._get_utility(network_bw, partition_idx, overall_thr_weight, mobile_thr_weight,
                                        CPU_freq, server_ratio, min_e2e, max_e2e, min_band, max_band)
            if max_utility is None or utility > max_utility:
                max_utility = utility
                max_idx = partition_idx
        return max_idx, max_utility

    def baseline_cloud_get_partition_point(self, network_bw, overall_thr_weight, mobile_thr_weight, CPU_freq,
                                           server_ratio):
        # 1. check the input parameters
        if overall_thr_weight < 0 or mobile_thr_weight < 0 or (overall_thr_weight + mobile_thr_weight) > 1:
            print("invalid weights", overall_thr_weight, mobile_thr_weight)
            return
        # 2. iterate the candidate partition layer
        max_utility, max_idx = None, 0
        min_e2e, max_e2e = self.get_max_min_e2e(network_bw, server_ratio)
        min_band, max_band = self.get_max_min_band(network_bw)
        partition_idx = 0
        utility = self._get_utility(network_bw, partition_idx, overall_thr_weight, mobile_thr_weight,
                                    CPU_freq, server_ratio, min_e2e, max_e2e, min_band, max_band)
        return partition_idx, utility

    def baseline_pi_get_partition_point(self, network_bw, overall_thr_weight, mobile_thr_weight, CPU_freq,
                                        server_ratio):
        # 1. check the input parameters
        if overall_thr_weight < 0 or mobile_thr_weight < 0 or (overall_thr_weight + mobile_thr_weight) > 1:
            print("invalid weights", overall_thr_weight, mobile_thr_weight)
            return
        # 2. iterate the candidate partition layer
        max_utility, max_idx = None, 0
        min_e2e, max_e2e = self.get_max_min_e2e(network_bw, server_ratio)
        min_band, max_band = self.get_max_min_band(network_bw)
        partition_idx = 8
        utility = self._get_utility(network_bw, partition_idx, overall_thr_weight, mobile_thr_weight,
                                    CPU_freq, server_ratio, min_e2e, max_e2e, min_band, max_band)
        return partition_idx, utility

    def _get_utility(self, network_bw, partition_idx, overall_thr_weight, mobile_thr_weight,
                     CPU_freq, server_ratio, min_e2e, max_e2e, min_band, max_band):

        latency_ratio = CPU_freq
        pi_latency, server_latency = self.pi_latency[partition_idx] * latency_ratio, \
                                     (self.server_latency[-1] - self.server_latency[
                                         partition_idx]) * latency_ratio * server_ratio

        band_occupation = self.layer_size[partition_idx] / (network_bw / 8.0) * 1 / min(33.33,
                                                                                        pi_latency)  # layer size in MB, network bandwidth as Mbps
        net_latency = self.layer_size[partition_idx] / (network_bw / 8.0) + 0.007

        # norm_e2e = (pi_latency + net_latency + server_latency - min_e2e)/(max_e2e-min_e2e)
        norm_e2e = (math.tanh(1 / (pi_latency + net_latency + server_latency)) - math.tanh(1 / max_e2e)) / (
                    math.tanh(1 / min_e2e) - math.tanh(1 / max_e2e))

        # norm_pi = (pi_latency-self.pi_latency[0])/(self.pi_latency[-1]-self.pi_latency[0])
        norm_pi = (math.tanh(1 / pi_latency) - math.tanh(1 / self.pi_latency[-1])) / (
                    math.tanh(1 / self.pi_latency[0]) - math.tanh(1 / self.pi_latency[-1]))

        norm_network = (math.tanh(1 / band_occupation) - math.tanh(1 / max_band)) / (
                    math.tanh(1 / min_band) - math.tanh(1 / max_band))

        # if round(net_latency, 4) == 0:
        #     net_latency = 0.0001

        utility = overall_thr_weight * norm_e2e + \
                  mobile_thr_weight * norm_pi + \
                  (1 - overall_thr_weight - mobile_thr_weight) * norm_network

        # print("layer name", self.layer_name[partition_idx])
        # print("overall", overall_thr_weight * math.tanh(1 / (pi_latency + net_latency + server_latency)))
        # print("pi", mobile_thr_weight * math.tanh(1 / pi_latency))
        # print("network", (1 - overall_thr_weight - mobile_thr_weight) * math.tanh(net_latency))
        # print("utility", utility)
        # print()
        return utility
