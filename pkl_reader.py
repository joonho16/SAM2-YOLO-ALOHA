import pickle

# pkl 파일 경로 설정
pkl_path = './ckpt/grasp_cable_test_ts/dataset_stats.pkl'

# pkl 파일 읽기
with open(pkl_path, 'rb') as file:
    data = pickle.load(file)

# 읽어온 데이터 출력 (데이터 구조에 따라 다를 수 있음)
print(data['action_std'])