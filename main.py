import time
from datetime import datetime
from argparse import ArgumentParser
from tool import set_seed
import yaml

from abfcm import abfcm_train_group, abfcm_output_group, abfcm_nms, abfcm_iou_process, \
    abfcm_final_result_per_subject, abfcm_final_result_best, abfcm_train_and_eval

def create_folder(opt):
    # create folder
    output_path = os.path.join(opt['project_root'], opt['output_dir_name'])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for subject in subject_list:
        out_subject_path = os.path.join(output_path, subject)
        if not os.path.exists(out_subject_path):
            os.mkdir(out_subject_path)
        subject_abfcm_out = os.path.join(out_subject_path, 'abfcm_out')
        if not os.path.exists(subject_abfcm_out):
            os.mkdir(subject_abfcm_out)
        subject_abfcm_nms_path = os.path.join(out_subject_path, 'abfcm_nms')
        if not os.path.exists(subject_abfcm_nms_path):
            os.mkdir(subject_abfcm_nms_path)
        subject_abfcm_final_result_path = os.path.join(
            out_subject_path, 'sub_abfcm_final_result')
        if not os.path.exists(subject_abfcm_final_result_path):
            os.mkdir(subject_abfcm_final_result_path)


def abfcm_train_mul_process(subject_group, opt):
    print("abfcm abfcm_train_mul_process ------ start: ")
    print("abfcm_training_lr: ", opt["abfcm_training_lr"])
    print("abfcm_weight_decay: ", opt["abfcm_weight_decay"])
    print("abfcm_lr_scheduler: ", opt["abfcm_lr_scheduler"])
    print("abfcm_apex_gamma: ", opt["abfcm_apex_gamma"])
    print("abfcm_apex_alpha: ", opt["abfcm_apex_alpha"])
    print("abfcm_action_gamma: ", opt["abfcm_action_gamma"])
    print("abfcm_action_alpha: ", opt["abfcm_action_alpha"])

    start_time = datetime.now()
    for subject_list in subject_group:
        abfcm_train_group(opt, subject_list)
        time.sleep(1)
    delta_time = datetime.now() - start_time
    print("abfcm abfcm_train_mul_process ------ sucessed: ")
    print("time: ", delta_time)


def abfcm_output_mul_process(subject_group, opt):
    print("abfcm_output_mul_process ------ start: ")
    print("micro_apex_score_threshold: ", opt["micro_apex_score_threshold"])
    print("macro_apex_score_threshold: ", opt["macro_apex_score_threshold"])
    start_time = datetime.now()
    for subject_list in subject_group:
        abfcm_output_group(opt, subject_list)
    delta_time = datetime.now() - start_time
    print("abfcm_output_mul_process ------ sucessed: ")
    print("time: ", delta_time)


def abfcm_nms_mul_process(subject_list, opt):
    print("abfcm_nms ------ start: ")
    print("nms_top_K: ", opt["nms_top_K"])
    start_time = datetime.now()
    for subject in subject_list:
        abfcm_nms(opt, subject)
    delta_time = datetime.now() - start_time
    print("abfcm_nms ------ sucessed: ")
    print("time: ", delta_time)


def abfcm_iou_mul_process(subject_list, opt):
    print("abfcm_iou_process ------ start: ")
    start_time = datetime.now()
    for subject in subject_list:
        abfcm_iou_process(opt, subject)
    delta_time = datetime.now() - start_time
    print("abfcm_iou_process ------ sucessed: ")
    print("time: ", delta_time)


if __name__ == "__main__":
    set_seed(270)

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = ArgumentParser()
    parser.add_argument("--mode", type=str, default='train_test', help='process, "train" or "test".')
    parser.add_argument('--dataset', type=str, default='cas(me)^2', help='dataset, "cas(me)^2", "samm", or "cas(me)^3".')
    parser.add_argument('--cuda', type=int, default=3)
    args = parser.parse_args()

    with open("./config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        if args.dataset is not None:
            dataset = args.dataset
        else:
            dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    subject_list = opt['subject_list']

    print("Config file loaded.")
    print("Validated subjects: " + str(subject_list))

    if args.mode is not None:
        opt["mode"] = args.mode
    opt['device'] = 'cuda:' + str(args.cuda)

    create_folder(opt)

    print(f"===================== Dataset is {dataset} =====================")

    if dataset != "cross":
        tmp_work_numbers = 1
        subject_group = []
        if len(subject_list) % tmp_work_numbers == 0:
            len_per_group = int(len(subject_list) // tmp_work_numbers)
            for i in range(tmp_work_numbers):
                subject_group.append(subject_list[
                                     i * len_per_group:
                                     (i + 1) * len_per_group])
        else:
            len_per_group = int(len(subject_list) // tmp_work_numbers) + 1
            last_len = len(subject_list) - len_per_group * (tmp_work_numbers - 1)
            for i in range(tmp_work_numbers - 1):
                subject_group.append(subject_list[
                                     i * len_per_group:
                                     (i + 1) * len_per_group])
            subject_group.append(subject_list[-last_len:])

        if opt["mode"] == "train_test":
            abfcm_train_mul_process(subject_group, opt)
            abfcm_output_mul_process(subject_group, opt)
            abfcm_nms_mul_process(subject_list, opt)
            abfcm_iou_mul_process(subject_list, opt)
            print("abfcm_final_result ------ start: ")
            # smic doesn't have macro labels
            if dataset != "smic":
                abfcm_final_result_per_subject(opt, subject_list, type_idx=1)
            abfcm_final_result_per_subject(opt, subject_list, type_idx=2)
            abfcm_final_result_per_subject(opt, subject_list, type_idx=0)
            abfcm_final_result_best(opt, subject_list, type_idx=0)
            print(opt['device'])
            print("abfcm_final_result ------ successed")
    else:
        abfcm_train_and_eval(opt)
