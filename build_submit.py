# -*- coding: utf-8 -*-
# @Time     : 2019/5/27 9:49
# @Author   : buracagyang
# @File     : build_submit.py
# @Software : PyCharm

"""
Describe:
        
"""

import json
import csv
import io
import os


def build():
    submit_map = {}
    if not os.path.isdir("./data/submit"):
        os.mkdir("./data/submit")

    with io.open('./data/submit/submit.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['sid', 'recommend_mode'])
        with open('./data/output/test_json_instance', 'r') as f1:
            with open('./data/output/predict_prob', 'r') as f2:
                cur_session = ''
                for x, y in zip(f1.readlines(), f2.readlines()):
                    m1 = json.loads(x)
                    session_id = m1["session_id"]
                    if cur_session == '':
                        cur_session = session_id

                    transport_mode = m1["plan"]["transport_mode"]

                    if cur_session != session_id:
                        writer.writerow([str(cur_session), str(submit_map[cur_session]["transport_mode"])])
                        cur_session = session_id

                    # 选择最大一个点击概率对应的model; 如果是第一次出现，则将数值直接写入submit_map
                    if session_id not in submit_map:
                        submit_map[session_id] = {}
                        submit_map[session_id]["transport_mode"] = transport_mode
                        submit_map[session_id]["probability"] = y

                    # 否则，需要比较然后选出最大的点击概率对应的交通模式
                    else:
                        if float(y) > float(submit_map[session_id]["probability"]):
                            submit_map[session_id]["transport_mode"] = transport_mode
                            submit_map[session_id]["probability"] = y

        writer.writerow([cur_session, submit_map[cur_session]["transport_mode"]])


if __name__ == "__main__":
    build()
