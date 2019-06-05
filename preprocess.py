# -*- coding: utf-8 -*-
# @Time     : 2019/5/26 15:45
# @Author   : buracagyang
# @File     : preprocess.py
# @Software : PyCharm


import os
import csv
import random
import datetime
import json
import pandas as pd


TRAIN_QUERIES_PATH = "./data/data_set_phase1/train_queries.csv"
TRAIN_PLANS_PATH = "./data/data_set_phase1/train_plans.csv"
TRAIN_CLICK_PATH = "./data/data_set_phase1/train_clicks.csv"
TEST_QUERIES_PATH = "./data/data_set_phase1/test_queries.csv"
TEST_PLANS_PATH = "./data/data_set_phase1/test_plans.csv"
PROFILES_PATH = "./data/data_set_phase1/profiles.csv"

WEATHER_PATH = "./data/extra_input_data/weather.json"

OUT_DIR = "./data/output"
ORI_TRAIN_PATH = "train.txt"
ORI_TEST_PATH = "test.txt"
ORI_TEST_JSON_PATH = 'test_json_instance'

HASH_DIM = 1000001

# 控制生成transport model 为0的比例
THRESHOLD_LABEL = 0.5

O1_MIN = 115.47
O1_MAX = 117.29

O2_MIN = 39.46
O2_MAX = 40.97

D1_MIN = 115.44
D1_MAX = 117.37

D2_MIN = 39.46
D2_MAX = 40.96

DISTANCE_MIN = 1.0
DISTANCE_MAX = 225864.0
DISTANCE_TS = 200000.0

PRICE_MIN = 200.0
PRICE_MAX = 92300.0
PRICE_TS = 20000

ETA_MIN = 1.0
ETA_MAX = 72992.0
ETA_TS = 10800.0

# 连续变量
continuous_features_list = ["distance", "price", "eta"]

# 分类变量
plan_feature_list = ["transport_mode"]
rank_feature_list = ["plan_rank", "whole_rank", "price_rank", "eta_rank", "distance_rank"]
rank_whole_pic_list = ["mode_rank1", "mode_rank2", "mode_rank3", "mode_rank4", "mode_rank5"]
query_feature_list = ["weekday", "hour", "o1", "o2", "d1", "d2"]
weather_feature_list = ["max_temp", "min_temp", "wea", "wind"]
pid_list = ['pid']  # 加入后效果貌还变差了...
profile_features_list = ['p{}'.format(i) for i in range(66)]  # 其实是已经one-hot后的特征


def preprocess_main():
    base_preprocess(TRAIN_QUERIES_PATH, TRAIN_PLANS_PATH, TRAIN_CLICK_PATH, train=True)
    base_preprocess(TEST_QUERIES_PATH, TEST_PLANS_PATH, TRAIN_CLICK_PATH, train=False)


def base_preprocess(queries_path, plans_path, clicks_path, train=True):

    train_data_dict = {}

    with open(WEATHER_PATH, 'r') as f:
        weather_dict = json.load(f)

    with open(queries_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for k, line in enumerate(csv_reader):
            if k == 0:
                continue
            if line[0] == "":
                continue

            train_index = line[0]
            train_data_dict[train_index] = {}
            train_data_dict[train_index]["pid"] = line[1]
            train_data_dict[train_index]["query"] = {}
            train_data_dict[train_index]["weather"] = {}

            reqweekday = datetime.datetime.strptime(line[2], '%Y-%m-%d %H:%M:%S').strftime("%w")
            reqhour = datetime.datetime.strptime(line[2], '%Y-%m-%d %H:%M:%S').strftime("%H")

            date_key = datetime.datetime.strptime(line[2], '%Y-%m-%d %H:%M:%S').strftime("%m-%d")
            train_data_dict[train_index]["weather"] = {}
            train_data_dict[train_index]["weather"].update({"max_temp": weather_dict[date_key]["max_temp"]})
            train_data_dict[train_index]["weather"].update({"min_temp": weather_dict[date_key]["min_temp"]})
            train_data_dict[train_index]["weather"].update({"wea": weather_dict[date_key]["weather"]})
            train_data_dict[train_index]["weather"].update({"wind": weather_dict[date_key]["wind"]})

            train_data_dict[train_index]["query"].update({"weekday": reqweekday})
            train_data_dict[train_index]["query"].update({"hour": reqhour})

            o = line[3].split(',')
            o_first = o[0]
            o_second = o[1]
            train_data_dict[train_index]["query"].update({"o1": float(o_first)})
            train_data_dict[train_index]["query"].update({"o2": float(o_second)})

            d = line[4].split(',')
            d_first = d[0]
            d_second = d[1]
            train_data_dict[train_index]["query"].update({"d1": float(d_first)})
            train_data_dict[train_index]["query"].update({"d2": float(d_second)})

    plan_map = {}
    plan_data = pd.read_csv(plans_path)
    for index, row in plan_data.iterrows():
        plans_str = row['plans']
        plans_list = json.loads(plans_str)
        session_id = str(row['sid'])
        plan_map[session_id] = plans_list

    profile_map = {}
    tmp_fea_name = None
    with open(PROFILES_PATH, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for k, line in enumerate(csv_reader):
            if k == 0:
                tmp_fea_name = line
                continue
            profile_map[line[0]] = dict(zip(tmp_fea_name[1:], line[1:]))

    session_click_map = {}
    with open(clicks_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for k, line in enumerate(csv_reader):
            if k == 0:
                continue
            if line[0] == "" or line[1] == "" or line[2] == "":
                continue
            session_click_map[line[0]] = line[2]
    if train:
        generate_sparse_features_train(train_data_dict, plan_map, profile_map, session_click_map)
    else:
        generate_sparse_features_test(train_data_dict, plan_map, profile_map)


def generate_sparse_features_train(train_data_dict, plan_map, profile_map, session_click_map):

    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)

    with open(os.path.join(OUT_DIR, ORI_TRAIN_PATH), 'w') as f_out:
        for session_id, plan_list in plan_map.items():
            if session_id not in train_data_dict:
                continue
            cur_map = train_data_dict[session_id]

            # 获取到用户配置文件的特征
            if cur_map["pid"] != "":
                cur_map["profile"] = profile_map[cur_map["pid"]]
            else:
                cur_map["profile"] = dict(zip(profile_features_list, [-1] * len(profile_features_list)))

            # rank information related feature
            whole_rank = 0
            for plan in plan_list:
                whole_rank += 1
                cur_map["mode_rank" + str(whole_rank)] = plan["transport_mode"]

            if whole_rank < 5:
                for r in range(whole_rank + 1, 6):
                    cur_map["mode_rank" + str(r)] = -1

            cur_map["whole_rank"] = whole_rank

            # 获取运输模式的price、eta、distance的统计信息
            price_list = []
            eta_list = []
            distance_list = []
            for plan in plan_list:
                if not plan["price"]:
                    price_list.append(0)
                else:
                    price_list.append(int(plan["price"]))
                eta_list.append(int(plan["eta"]))
                distance_list.append(int(plan["distance"]))
            price_list.sort(reverse=False)
            eta_list.sort(reverse=False)
            distance_list.sort(reverse=False)

            for plan in plan_list:
                if plan["price"] and int(plan["price"]) == price_list[0]:
                    cur_map["mode_min_price"] = plan["transport_mode"]
                if plan["price"] and int(plan["price"]) == price_list[-1]:
                    cur_map["mode_max_price"] = plan["transport_mode"]
                if int(plan["eta"]) == eta_list[0]:
                    cur_map["mode_min_eta"] = plan["transport_mode"]
                if int(plan["eta"]) == eta_list[-1]:
                    cur_map["mode_max_eta"] = plan["transport_mode"]
                if int(plan["distance"]) == distance_list[0]:
                    cur_map["mode_min_distance"] = plan["transport_mode"]
                if int(plan["distance"]) == distance_list[-1]:
                    cur_map["mode_max_distance"] = plan["transport_mode"]
            if "mode_min_price" not in cur_map:
                cur_map["mode_min_price"] = -1
            if "mode_max_price" not in cur_map:
                cur_map["mode_max_price"] = -1

            # 确定运输模式是否被点击
            flag_click = False
            rank = 1
            for plan in plan_list:
                if ("transport_mode" in plan) and (session_id in session_click_map) and \
                        (int(plan["transport_mode"]) == int(session_click_map[session_id])):
                    flag_click = True
                    break
            if flag_click:
                for plan in plan_list:
                    cur_price = int(plan["price"]) if plan["price"] else 0
                    cur_eta = int(plan["eta"])
                    cur_distance = int(plan["distance"])
                    cur_map["price_rank"] = price_list.index(cur_price) + 1
                    cur_map["eta_rank"] = eta_list.index(cur_eta) + 1
                    cur_map["distance_rank"] = distance_list.index(cur_distance) + 1

                    # 点击的运输模式，label赋值为1；其他，label赋值为0
                    if ("transport_mode" in plan) and (session_id in session_click_map) and \
                            (int(plan["transport_mode"]) == int(session_click_map[session_id])):
                        cur_map["plan"] = plan
                        cur_map["label"] = 1
                    else:
                        cur_map["plan"] = plan
                        cur_map["label"] = 0

                    cur_map["plan_rank"] = rank
                    rank += 1

                    cleared_instance_str = get_cleared_instance(cur_map)
                    f_out.write(cleared_instance_str)
                # generate model 0
                if random.random() < THRESHOLD_LABEL:
                    cur_map["plan"]["distance"] = -1
                    cur_map["plan"]["price"] = -1
                    cur_map["plan"]["eta"] = -1
                    cur_map["plan"]["transport_mode"] = 0
                    cur_map["plan_rank"] = 0
                    cur_map["distance_rank"] = 0
                    cur_map["price_rank"] = 0
                    cur_map["eta_rank"] = 0
                    cur_map["label"] = 0
                    cleared_instance_str = get_cleared_instance(cur_map)
                    f_out.write(cleared_instance_str)

            cur_map["plan"] = {}
            if not flag_click:
                # for plan in plan_list:
                #     cur_price = int(plan["price"]) if plan["price"] else 0
                #     cur_eta = int(plan["eta"])
                #     cur_distance = int(plan["distance"])
                #     cur_map["price_rank"] = price_list.index(cur_price) + 1
                #     cur_map["eta_rank"] = eta_list.index(cur_eta) + 1
                #     cur_map["distance_rank"] = distance_list.index(cur_distance) + 1
                #     cur_map["plan"] = plan
                #     cur_map["label"] = 0
                #     cur_map["plan_rank"] = rank
                #     rank += 1
                #     cleared_instance_str = get_cleared_instance(cur_map)
                #     f_out.write(cleared_instance_str)
                # generate model 0
                cur_map["plan"]["distance"] = -1
                cur_map["plan"]["price"] = -1
                cur_map["plan"]["eta"] = -1
                cur_map["plan"]["transport_mode"] = 0
                cur_map["plan_rank"] = 0
                cur_map["distance_rank"] = 0
                cur_map["price_rank"] = 0
                cur_map["eta_rank"] = 0
                cur_map["label"] = 1
                cleared_instance_str = get_cleared_instance(cur_map)
                f_out.write(cleared_instance_str)


def generate_sparse_features_test(train_data_dict, plan_map, profile_map):
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)

    with open(os.path.join(OUT_DIR, ORI_TEST_PATH), 'w') as f_out:
        with open(os.path.join(OUT_DIR, ORI_TEST_JSON_PATH), 'w') as json_f_out:
            for session_id, plan_list in plan_map.items():
                if session_id not in train_data_dict:
                    continue
                cur_map = train_data_dict[session_id]
                cur_map["session_id"] = session_id
                if cur_map["pid"] != "":
                    cur_map["profile"] = profile_map[cur_map["pid"]]
                else:
                    cur_map["profile"] = dict(zip(profile_features_list, [-1] * len(profile_features_list)))
                whole_rank = 0
                for plan in plan_list:
                    whole_rank += 1
                    cur_map["mode_rank" + str(whole_rank)] = plan["transport_mode"]
                if whole_rank < 5:
                    for r in range(whole_rank + 1, 6):
                        cur_map["mode_rank" + str(r)] = -1
                cur_map["whole_rank"] = whole_rank

                price_list = []
                eta_list = []
                distance_list = []
                for plan in plan_list:
                    if not plan["price"]:
                        price_list.append(0)
                    else:
                        price_list.append(int(plan["price"]))
                    eta_list.append(int(plan["eta"]))
                    distance_list.append(int(plan["distance"]))
                price_list.sort(reverse=False)
                eta_list.sort(reverse=False)
                distance_list.sort(reverse=False)

                for plan in plan_list:
                    if plan["price"] and int(plan["price"]) == price_list[0]:
                        cur_map["mode_min_price"] = plan["transport_mode"]
                    if plan["price"] and int(plan["price"]) == price_list[-1]:
                        cur_map["mode_max_price"] = plan["transport_mode"]
                    if int(plan["eta"]) == eta_list[0]:
                        cur_map["mode_min_eta"] = plan["transport_mode"]
                    if int(plan["eta"]) == eta_list[-1]:
                        cur_map["mode_max_eta"] = plan["transport_mode"]
                    if int(plan["distance"]) == distance_list[0]:
                        cur_map["mode_min_distance"] = plan["transport_mode"]
                    if int(plan["distance"]) == distance_list[-1]:
                        cur_map["mode_max_distance"] = plan["transport_mode"]
                if "mode_min_price" not in cur_map:
                    cur_map["mode_min_price"] = -1
                if "mode_max_price" not in cur_map:
                    cur_map["mode_max_price"] = -1

                rank = 1
                for plan in plan_list:
                    cur_price = int(plan["price"]) if plan["price"] else 0
                    cur_eta = int(plan["eta"])
                    cur_distance = int(plan["distance"])
                    cur_map["price_rank"] = price_list.index(cur_price) + 1
                    cur_map["eta_rank"] = eta_list.index(cur_eta) + 1
                    cur_map["distance_rank"] = distance_list.index(cur_distance) + 1
                    # 不会出现以下分支
                    # if ("transport_mode" in plan) and (session_id in session_click_map) \
                    #         and (int(plan["transport_mode"]) == int(session_click_map[session_id])):
                    #     cur_map["plan"] = plan
                    #     cur_map["label"] = 1
                    # else:
                    #     cur_map["plan"] = plan
                    #     cur_map["label"] = 0
                    cur_map["plan"] = plan
                    cur_map["label"] = 0

                    cur_map["plan_rank"] = rank
                    rank += 1
                    json_f_out.write(json.dumps(cur_map) + '\n')
                    cleared_instance_str = get_cleared_instance(cur_map)
                    f_out.write(cleared_instance_str)

                # ###################################################
                # 构建一个transport_model为0的样例
                # ###################################################
                cur_map["plan"]["distance"] = -1
                cur_map["plan"]["price"] = -1
                cur_map["plan"]["eta"] = -1
                cur_map["plan"]["transport_mode"] = 0
                cur_map["plan_rank"] = 0
                cur_map["distance_rank"] = 0
                cur_map["distance_rank"] = 0
                cur_map["price_rank"] = 0
                cur_map["eta_rank"] = 0
                cur_map["label"] = 1
                json_f_out.write(json.dumps(cur_map) + '\n')

                cleared_instance_str = get_cleared_instance(cur_map)
                f_out.write(cleared_instance_str)


def get_cleared_instance(cur_instance):
    label = str(cur_instance['label'])

    all_features_list = []
    for fea in continuous_features_list:
        all_features_list.append(str(cur_instance['plan'][fea]))
    for fea in plan_feature_list:
        all_features_list.append(str(cur_instance['plan'][fea]))
    for fea in rank_feature_list:
        all_features_list.append(str(cur_instance[fea]))
    for fea in rank_whole_pic_list:
        all_features_list.append(str(cur_instance[fea]))
    for fea in query_feature_list:
        all_features_list.append(str(cur_instance['query'][fea]))
    for fea in weather_feature_list:
        all_features_list.append(str(cur_instance['weather'][fea]))
    for fea in pid_list:
        all_features_list.append(str(cur_instance[fea]))
    for fea in profile_features_list:
        all_features_list.append(str(cur_instance['profile'][fea]))
    # for fea in mode_rank_list:
    #     all_features_list.append(str(cur_instance[fea]))

    return "{0} {1}\n".format(label, ' '.join(all_features_list))


if __name__ == "__main__":
    preprocess_main()
