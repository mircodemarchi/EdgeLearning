import numpy as np
import pandas as pd
import math
import time

class DlMath: 
    @staticmethod
    def elu(x, alpha=1.0):
        return x if x > 0 else alpha * (math.exp(x) - 1)

    @staticmethod
    def elu_1(x, alpha=1.0):
        return 1 if x > 0 else alpha * math.exp(x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_1(x):
        return DlMath.sigmoid(x) * (1 - DlMath.sigmoid(x))

    @staticmethod
    def dense(x, w, b):
        return w.dot(x) + b

    @staticmethod
    def dense_1(x, last_input, w):
        x_1 = w.transpose().dot(x)
        b_1 = x
        w_1 = x[:, np.newaxis].dot(last_input[np.newaxis, :])
        return (x_1, w_1, b_1)


class TestDlMath: 
    @staticmethod
    def run():
        TestDlMath.elu()
        TestDlMath.sigmoid()
        TestDlMath.dense()

    @staticmethod
    def elu():
        elu_in = [-2,-1,0,1,2]
        print("elu: {}".format([DlMath.elu(e) for e in elu_in]))

        elu_1_in = [-2,-1,0,1,2]
        print("elu_1: {}".format([DlMath.elu_1(e) for e in elu_1_in]))

    @staticmethod
    def sigmoid():
        sigmoid_in = [-10.0, 0.0, 1.0, 7.0, 10000.0]
        print("sigmoid: {}".format([DlMath.sigmoid(e) for e in sigmoid_in]))

        sigmoid_1_in = [-10.0, 0.0, 1.0, 7.0, 10000.0]
        print("sigmoid_1: {}".format([DlMath.sigmoid_1(e) for e in sigmoid_1_in]))

    @staticmethod
    def dense():
        dense_w = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8]
        ])
        dense_b = np.array([0.1, 0.2])
        dense_in = np.array([1,2,3,4])

        print("dense: {}".format(DlMath.dense(dense_in, dense_w, dense_b)))

        dense_g = np.array([-0.1, -0.2])
        dense_in_1, dense_w_1, dense_b_1 = DlMath.dense_1(dense_g, dense_in, dense_w)
        print("dense_1: {{ \n\tin_1: {}, \n\tw_1: {}, \n\tb_1: {} \n}}".format(dense_in_1, dense_w_1, dense_b_1))


class ProfileDlMath: 
    DENSE_PARAMS = [
        (10,    10),
        (10,    100),
        (100,   100),
        (100,   1000),
        (1000,  1000),
        (1000,  10000),
        (10000, 10000)
    ] 

    @staticmethod
    def pretty_print(cnt):
        if cnt < 1000:
            return "{} ns".format(cnt)
        elif cnt < 1000000:
            return "{} us".format(cnt / 1000)
        elif cnt < 1000000000:
            return "{} ms".format(cnt / 1000000)
        elif cnt < 1000000000000:
            return "{} sec".format(cnt / 1000000000)
        else:
            return "{} min".format(cnt / 60000000000)

    @staticmethod
    def run(iter = 100):
        ProfileDlMath.dense(iter)

    @staticmethod
    def dense(iter):
        time_results = {}
        for dp in ProfileDlMath.DENSE_PARAMS:
            x = (np.random.rand(dp[0]) * 2 - 1) * 10
            w = (np.random.rand(dp[1], dp[0]) * 2 - 1) * 10
            b = (np.random.rand(dp[1]) * 2 - 1) * 10

            times = []
            for _ in range(iter):
                t0 = int(time.time() * 1e9)
                DlMath.dense(x, w, b)
                t1 = int(time.time() * 1e9)
                times.append(t1 - t0)

            time_results["dense_on_{}_{}x{}".format("numpy", dp[0], dp[1])] = times

        for l in time_results: 
            times = time_results[l]
            times_np = np.array(times)
            print("{}: mean: {} median: {} std: {}".format(
                l, 
                ProfileDlMath.pretty_print(np.mean(times_np)), 
                ProfileDlMath.pretty_print(np.median(times_np)), 
                ProfileDlMath.pretty_print(np.std(times_np))
            ))

        for l in time_results: 
            times = time_results[l]
            times_df = pd.DataFrame({"time": times})
            times_df.to_csv("{}.csv".format(l))

    @staticmethod
    def dense_1(iter):
        time_results = {}
        for dp in ProfileDlMath.DENSE_PARAMS:
            x = (np.random.rand(dp[1]) * 2 - 1) * 10
            last_input = (np.random.rand(dp[0]) * 2 - 1) * 10
            w = (np.random.rand(dp[1], dp[0]) * 2 - 1) * 10

            times = []
            for _ in range(iter):
                t0 = int(time.time() * 1e9)
                DlMath.dense_1(x, last_input, w)
                t1 = int(time.time() * 1e9)
                times.append(t1 - t0)

            time_results["dense_1_on_{}_{}x{}".format("numpy", dp[0], dp[1])] = times
        
        for l in time_results: 
            times = time_results[l]
            times_np = np.array(times)
            print("{}: mean: {} median: {} std: {}".format(
                l, 
                ProfileDlMath.pretty_print(np.mean(times_np)), 
                ProfileDlMath.pretty_print(np.median(times_np)), 
                ProfileDlMath.pretty_print(np.std(times_np))
            ))
        
        for l in time_results: 
            times = time_results[l]
            times_df = pd.DataFrame({"time": times})
            times_df.to_csv("{}.csv".format(l))
            

TestDlMath.run()
ProfileDlMath.run()