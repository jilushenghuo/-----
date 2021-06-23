#-*-coding:gb2312-*-
import os
# class TestAccuracyRate():
import cv2
import face_recognition
import numpy as np


def trainImage():
    train_dir = 'rate_train_data'
    # ѵ���ļ���
    # ��ѵ���ļ����µ�ͼƬ�ŵ�һ���б���
    train_image_list = os.listdir(train_dir)
    print("ѵ��ͼƬ������%s" % len(train_image_list))

    # ������б�����Ķ�ȡ��ͼƬ����
    train_image_names_list = []
    # ��Ŷ�ȡ��ͼƬ���·��
    train_image_paths_list = []
    # ���ͼƬת����numpy����
    train_image_rec_list = []
    # ���ÿ��ͼƬ�沿����ĵ�һ���沿������Ϣ
    train_image_encodings_list = []
    'Դ�ļ���ȡѵ������'
    # �����б���ȡ�б��е�ѵ��ͼƬ�ļ���
    for per_image in train_image_list:
        name = per_image.strip('.jpg')
        # print(name)
        train_image_names_list.append(name)  # ����ȡ��ͼƬ������ӵ�Ԥ�ȶ���õ��б���
        # �趨ÿ��ͼƬ�����·��������·����ȡÿ��ͼƬ
        train_per_image_path = os.path.join(train_dir, per_image)
        train_image_paths_list.append(train_per_image_path)  # ��ÿ��ͼƬ��·����ӵ�Ԥ�ȶ���õ��б���
        # ����ѵ��ͼ��ͨ��face_recognitionѧϰ���ʶ����,���ĵ����ص�numpy������,�Ա���������ʶ��
        train_per_image_rec = face_recognition.load_image_file(train_per_image_path)
        train_image_rec_list.append(train_per_image_rec)
        print(train_per_image_rec[0])
        # ��ȡÿ��ͼ���沿������Ϣ
        train_per_image_encoding = face_recognition.face_encodings(train_per_image_rec)[0]
        print("ѵ��ͼƬ%s����������"%name)
        train_image_encodings_list.append(train_per_image_encoding)
        # print(train_per_image_encoding)
    # print(train_image_names_list)
    return train_image_names_list, train_image_encodings_list


def testImage():
    # '���в����ļ����Զ���ȡ'
    # ����ͼƬ·��
    test_dir = 'rate_test_data'  # �����ļ���
    # ������ͼƬ���ƶ�ȡ��һ���б���
    test_image_list = os.listdir(test_dir)
    # print(test_image_list)
    # ��ȡ����ͼ��������
    test_date_num = len(test_image_list)
    print('����ͼƬ����Ϊ��%s ��' % test_date_num)
    # ��ȡѵ������
    train_image_names_list, train_image_encodings_list = trainImage()
    print('========================')
    print("Դ�ļ���ȡ��ɡ�����")
    # ���ó�ʼ����ʶ��ɹ�����Ϊ0
    reco_count = 0  # �ɹ�ʶ������ʶ����
    decate_count = 0  # δ�ɹ�ʶ������ͼƬ��
    right_reco_count = 0  # ��ȷƥ������ͼƬ��
    undecate_count = 0  # δ�ɹ���⵽��������
    unsuccess_reco_count = 0  # δ�ɹ�ʶ������������
    unsuccess_reco_name_count = []  # δ�ɹ�ʶ�������б�
    undecate_name_list = []  # δ�ɹ���������б�
    false_right_reco_name_list = []  # ʶ��ƥ��ɹ�����δ��ȷƥ������ͼƬ�б�
    print("ͼ��ʶ���С�����")
    # �����б����в���ͼƬ����
    for per_image in test_image_list:

        # ͼ����������������������ȡ������
        name = per_image.strip('.jpg')
        # ͼ������·��
        test_per_image_path = os.path.join(test_dir, per_image)
        per_image_cv = cv2.imread(test_per_image_path)
        per_image_cvcolor = cv2.cvtColor(per_image_cv, cv2.COLOR_BGR2RGB)
        '�˴�ͼƬû�м�⵽����ʱ����ִ����ᱨ����Ҫ����һ��֮ǰ�����ж��������Ƿ��⵽������Ϣ��'
        per_image_encoding = face_recognition.face_encodings(per_image_cvcolor)
        # print(per_image_encoding)
        '�ж�������ͼƬ���������Ƿ�Ϊ�գ�'
        if len(per_image_encoding) > 0:
            decate_count += 1
            # ��ȡ��⵽����ʱ�沿������Ϣ������0λ���沿����
            per_image_encoding = per_image_encoding[0]
            "ͼ������ʶ�𲿷ִ��룬ͬʱʹ��compare_faces��face_distance�������ѵ�����׼ȷ�ȡ�"
            # �����沿����ƥ���������������б�
            matchs_bool_list = face_recognition.compare_faces(train_image_encodings_list, per_image_encoding,                                                tolerance=0.46)
            # print(matchs_bool_list)
            # print(train_image_names_list)
            # �������֮���ŷ�Ͼ��룬����һ����ֵ�б�
            face_distances_list = face_recognition.face_distance(train_image_encodings_list, per_image_encoding)
            # ����ŷʽ���룬������������׵�����
            # print(face_distances_list)
            best_match_index = np.argmin(face_distances_list)
            # print(best_match_index)
            # print(train_image_names_list[best_match_index])
            print("name:%s,match_name:%s" % (name, train_image_names_list[best_match_index]))
            if matchs_bool_list[best_match_index]:
                match_name = train_image_names_list[best_match_index]
                # best_match_encoding = train_image_encodings_list[best_match_index]
                print('ƥ������������%s' % match_name)
                reco_count += 1
            else:
                unsuccess_reco_count += 1
                unsuccess_reco_name = name + '.jpg'
                unsuccess_reco_name_count.append(unsuccess_reco_name)
                match_name = 'unknown_person'
            # �����ļ�����ͬ�������ļ�������׼ȷʶ��
            if name == match_name:
                right_reco_count += 1
            else:
                false_right_reco_name = name + '.jpg'
                false_right_reco_name_list.append(false_right_reco_name)
        else:
            undecate_count += 1
            undecate_name = name + '.jpg'
            undecate_name_list.append(undecate_name)
            print("ͼƬ%s.jpgδ��⵽��Ч��������,�����ϴ�ͼƬ�Ƿ�Ϊ������������" % name)
    print("ʶ����ɣ�")

    print("�ɹ��������ͼƬ������%s,δ��⵽����ͼƬ������%s || �ɹ�ʶ��ɹ�ƥ����������:%s,�ɹ�ʶ��δ�ɹ�ƥ�䵽����ͼƬ����%s || ��ȷƥ������ͼƬ������%s" % (
    decate_count, undecate_count, reco_count, unsuccess_reco_count, right_reco_count))
    print("�ɹ�ʶ��δ�ɹ�ƥ�䵽����ͼƬ�����б�%s" % (unsuccess_reco_name_count))
    print("ʶ��ƥ��ɹ�����δ��ȷƥ������ͼƬ�����б�%s" % (false_right_reco_name_list))
    print("δ�ɹ��������ͼƬ�����б�%s" % (undecate_name_list))
    reco_rate = reco_count / (test_date_num - undecate_count)
    right_reco_rate = right_reco_count / (test_date_num - undecate_count)
    print("ʶ����ɣ��������ɹ���Ϊ��%s%%,ʶ��׼ȷ��Ϊ��%s%%" % ((reco_rate * 100), (right_reco_rate * 100)))  # ʶ��׼ȷ�ʰٷֺ���ʾ
# cv2.destroyAllWindows()

if __name__ == '__main__':
    testImage()
