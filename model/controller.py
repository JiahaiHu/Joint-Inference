import math
import pandas as pd

from model.model_zoo.resnet_local import resnet50

class Code2SpacePartition:
    def __init__(self, model_name="resnet_50"):
        model_info = pd.read_excel("./model_zoo/model_info.xlsx", sheet_name=model_name)
        self.layer_name = model_info['layer_name'].values
        self.layer_size = model_info['layer_size'].values  # output size in MB
        self.pi_latency = model_info['pi_latency'].values
        self.server_latency = model_info['server_latency'].values

    def init_model(self, network_bw, overall_thr_weight=0.4, mobile_thr_weight=0.2, CPU_freq=1):
        # 1. get partition layer
        partition_idx, tmp = self.get_partition_point(network_bw, overall_thr_weight, mobile_thr_weight, CPU_freq)

        # 2. depend on partition idx to initialize the model instance
        if partition_idx == len(self.layer_name):
            return None
        else:
            net = resnet50(end_layer=self.layer_name[partition_idx])
        return net, self.layer_name[partition_idx]

    def predict_model(self):
        pass
        # net = self.init_model()
        # 1. receive data from pi
        # 1)numpy => ms.Tensor(npy_arr,dtype=ms.float32)
        # 2) run model => result = net(ms.Tensor(npy_arr,dtype=ms.float32))

    def get_partition_point(self, network_bw, overall_thr_weight, mobile_thr_weight, CPU_freq):
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
        for partition_idx, partition_layer in enumerate(self.layer_name):
            utility = self._get_utility(network_bw, partition_idx, overall_thr_weight, mobile_thr_weight, CPU_freq)
            if max_utility is None or utility > max_utility:
                max_utility = utility
                max_idx = partition_idx

        return max_idx, max_utility

    def _get_utility(self, network_bw, partition_idx, overall_thr_weight, mobile_thr_weight, CPU_freq):
        net_latency = self.layer_size[partition_idx] / (network_bw / 8.0)  # layer size in MB, network bandwidth as Mbps
        latency_ratio = CPU_freq
        pi_latency, server_latency = self.pi_latency[partition_idx] * latency_ratio, \
                                     (self.server_latency[-1] - self.server_latency[partition_idx]) * latency_ratio

        if round(net_latency, 4) == 0:
            net_latency = 0.0001

        utility = overall_thr_weight * math.tanh(1 / (pi_latency + net_latency + server_latency)) + \
                  mobile_thr_weight * math.tanh(1 / pi_latency) + \
                  (1 - overall_thr_weight - mobile_thr_weight) * math.tanh(net_latency)

        # print("layer name", self.layer_name[partition_idx])
        # print("overall", overall_thr_weight * math.tanh(1 / (pi_latency + net_latency + server_latency)))
        # print("pi", mobile_thr_weight * math.tanh(1 / pi_latency))
        # print("network", (1 - overall_thr_weight - mobile_thr_weight) * math.tanh(net_latency))
        # print("utility", utility)
        # print()
        return utility
