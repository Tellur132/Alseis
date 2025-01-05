import random
import cirq
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.linalg import eigh

CONFIG_PATH = 'config.yaml'


def load_config(config_path):
    """
    YAML形式の設定ファイルを読み込む関数。
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def set_random_seed(config):
    """
    設定に基づいてシードを固定またはランダムに設定します。
    """
    seed_config = config.get('random_seed', {})
    mode = seed_config.get('mode', 'random')

    if mode == 'fixed':
        seed = seed_config.get('value', 42)
        np.random.seed(seed)
        random.seed(seed)
        # Cirqでは明示的なシード設定は少ないが、シミュレーターにシードを渡すことが可能
        return seed
    elif mode == 'random':
        # シードを固定しない場合
        return None
    else:
        raise ValueError("random_seed.mode は 'fixed' または 'random' を指定してください。")


def load_and_scale_dataset(dataset_name, method='standard', random_state=None):
    """
    指定されたデータセットを読み込み、指定方法でスケーリングした上で返す関数。
    dataset_name: 'iris', 'wine', 'breast_cancer', 'digits', 'diabetes', 'moons', 'circles', 'blobs', 'high_dim' などを想定
    method: 'standard', 'minmax', 'robust'
    random_state: データ生成時のランダムシード
    """
    if dataset_name == 'iris':
        data = load_iris()
    elif dataset_name == 'wine':
        data = load_wine()
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    elif dataset_name == 'digits':
        data = load_digits()
    elif dataset_name == 'diabetes':
        data = load_diabetes()
    elif dataset_name == 'moons':
        data = make_moons(n_samples=500, noise=0.05, random_state=random_state)
    elif dataset_name == 'circles':
        data = make_circles(n_samples=500, noise=0.05,
                            factor=0.5, random_state=random_state)
    elif dataset_name == 'blobs':
        data = make_blobs(n_samples=500, centers=3,
                          cluster_std=1.0, random_state=random_state)
    elif dataset_name == 'high_dim':
        data = make_classification(n_samples=500, n_features=10,
                                   n_informative=8, n_redundant=2,
                                   n_classes=2, random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # data が Bunch であれば data.data でアクセスできる
    # タプルであれば (features, target) のような形式になっているので、
    # features = data[0] から取得
    if hasattr(data, 'data'):
        features = data.data
    else:
        # data がタプルだった場合
        features = data[0]

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown method: {method}")

    scaled_features = scaler.fit_transform(features)
    return scaled_features


def create_quantum_encoding_circuit(qubits, data, method='amplitude'):
    """
    データを量子状態にエンコードするための回路を作成します。
    method: 'amplitude', 'phase', 'angle', 'qrac' のいずれかを指定
    """
    circuit = cirq.Circuit()
    if method == 'amplitude':
        for i, value in enumerate(data):
            value = (value - np.min(data)) / \
                (np.max(data) - np.min(data) + 1e-10) * 2 - 1.0
            value = np.nan_to_num(value)
            angle = 2 * np.arccos(value)
            if i < len(qubits):
                circuit.append(cirq.ry(angle)(qubits[i]))
    elif method == 'phase':
        for i, value in enumerate(data):
            value = (value - np.min(data)) / \
                (np.max(data) - np.min(data) + 1e-10) * 2 - 1.0
            value = np.nan_to_num(value)
            angle = 2 * np.pi * value  # 位相を調整するために2πを掛ける
            if i < len(qubits):
                circuit.append(cirq.rz(angle)(qubits[i]))
    elif method == 'angle':
        for i, value in enumerate(data):
            angle = (value - np.min(data)) / \
                (np.max(data) - np.min(data) + 1e-10) * np.pi
            if i < len(qubits):
                circuit.append(cirq.rx(angle)(qubits[i]))
    elif method == 'qrac':
        # QRACを使ったエンコード
        # QRACは通常、2つの古典ビットを1つの量子ビットにエンコードすることに用いられます
        # ここでは、データを2つずつ取り出して1つの量子ビットにエンコードする
        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                # 2つの値を取り出し、QRACエンコード
                x1, x2 = data[i], data[i + 1]
                # 値を -1.0 から 1.0 に正規化
                x1 = (x1 - np.min(data)) / \
                    (np.max(data) - np.min(data) + 1e-10) * 2 - 1.0
                x2 = (x2 - np.min(data)) / \
                    (np.max(data) - np.min(data) + 1e-10) * 2 - 1.0
                x1, x2 = np.nan_to_num(x1), np.nan_to_num(x2)
                # QRACの回路として、例えばRyとRzを使用（簡易的な例）
                angle_y = np.arccos(x1)
                angle_z = np.arccos(x2)
                if i // 2 < len(qubits):
                    circuit.append(cirq.ry(2 * angle_y)(qubits[i // 2]))
                    circuit.append(cirq.rz(2 * angle_z)(qubits[i // 2]))
            elif i < len(data):
                # 最後の要素が余る場合、単独でエンコード
                value = (data[i] - np.min(data)) / \
                    (np.max(data) - np.min(data) + 1e-10) * 2 - 1.0
                value = np.nan_to_num(value)
                angle = 2 * np.arccos(value)
                if i // 2 < len(qubits):
                    circuit.append(cirq.ry(angle)(qubits[i // 2]))
    else:
        raise ValueError(
            "Invalid encoding method. Choose 'amplitude', 'phase', 'angle', or 'qrac'.")
    return circuit


def perform_qpca(circuits, num_iterations, simulator):
    """
    qPCAを実施し、エンコードされた量子データに対して次元削減を行います。
    """
    config = load_config(CONFIG_PATH)
    config_print_eigenvalues = config.get('print_eigenvalues', False)
    config_print_eigenvectors = config.get('print_eigenvectors', False)

    density_matrices = []
    for circuit in circuits:
        state = simulator.simulate(circuit)
        density_matrix = np.outer(
            state.final_state_vector, np.conj(state.final_state_vector))
        density_matrices.append(density_matrix)

    # 全ての密度行列を平均して最終的な密度行列を作成
    avg_density_matrix = np.mean(density_matrices, axis=0)

    # 固有値と固有ベクトルを計算（qPCAの中心的な部分）
    eigenvalues, eigenvectors = eigh(avg_density_matrix)

    if config_print_eigenvalues:
        print("Eigenvalues:")
        print(eigenvalues)
    if config_print_eigenvectors:
        print("Eigenvectors:")
        print(eigenvectors)

    # 固有値が大きいものに基づいて次元削減
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:num_iterations]]

    return top_eigenvectors, eigenvalues, sorted_indices


def decode_quantum_data(top_eigenvectors, scaled_data):
    """
    qPCAで次元削減した量子データを古典データに低次元空間に射影します
    """
    # 元のデータを低次元空間に射影するために、データと固有ベクトルの次元を整合させる
    top_eigenvectors_reduced = top_eigenvectors[:scaled_data.shape[1], :]
    reduced_data = np.dot(scaled_data, top_eigenvectors_reduced)
    return reduced_data


def calculate_contribution_ratios(eigenvalues, sorted_indices, num_components):
    """
    寄与率を計算し、各主成分の寄与率の表を作成します。
    """
    total_variance = np.sum(eigenvalues)
    selected_eigenvalues = eigenvalues[sorted_indices[:num_components]]
    contribution_ratios = selected_eigenvalues / total_variance
    contribution_table = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(num_components)],
        'Eigenvalue': selected_eigenvalues,
        'Contribution Ratio': contribution_ratios
    })
    return contribution_table


def main():
    # 設定ファイルのパスを指定
    config = load_config(CONFIG_PATH)

    # シードの設定
    seed = set_random_seed(config)

    dataset_names = config.get('dataset_name', 'iris').split(
        ',')   # ここを変更: データセット名を複数取得
    normalization_methods = config.get(
        'normalization_method', 'standard').split(',')
    encoding_methods = config.get('encoding_method', 'amplitude').split(',')
    config_print_circuit = config.get('print_circuit', False)
    config_print_top_eigenvectors = config.get('print_top_eigenvectors', False)
    config_print_classical_data = config.get('print_classical_data', False)

    # Cirqのシミュレーターにシードを渡す（固定シードが必要な場合）
    if seed is not None:
        simulator = cirq.Simulator(seed=seed)
    else:
        simulator = cirq.Simulator()

    for dname in dataset_names:
        # データセットごとに処理
        print(f"=== Dataset: {dname} ===")
        for norm_method in normalization_methods:
            print(f"Testing with {norm_method} normalization:")
            scaled_data = load_and_scale_dataset(
                dname.strip(), norm_method, random_state=seed if seed is not None else None)  # シードを渡す
            # 量子ビット準備
            num_qubits = scaled_data.shape[1]
            qubits = [cirq.LineQubit(i) for i in range(num_qubits)]

            for method in encoding_methods:
                print(f"Testing with {
                      method} encoding (Normalization: {norm_method}):")
                circuits = []
                for data in scaled_data:
                    circuits.append(
                        create_quantum_encoding_circuit(qubits, data, method))
                    for _ in range(5):
                        if seed is not None:
                            # シードが固定されている場合、ノイズ生成も固定
                            noise = np.random.default_rng(
                                seed).normal(0, 0.005, data.shape)
                        else:
                            noise = np.random.normal(0, 0.005, data.shape)
                        noisy_data = data + noise
                        noisy_data = np.clip(noisy_data, -1.0, 1.0)
                        noisy_data = np.nan_to_num(noisy_data)
                        circuits.append(create_quantum_encoding_circuit(
                            qubits, noisy_data, method))

                if config_print_circuit:
                    for i, circuit in enumerate(circuits):
                        print(f"Quantum Encoding Circuit for sample {i}:")
                        print(circuit)

                top_eigenvectors, eigenvalues, sorted_indices = perform_qpca(
                    circuits, num_iterations=2, simulator=simulator)
                if config_print_top_eigenvectors:
                    print("Top Eigenvectors after qPCA:")
                    print(top_eigenvectors)

                decoded_data = decode_quantum_data(
                    top_eigenvectors, scaled_data)
                if config_print_classical_data:
                    print("Decoded Classical Data:")
                    print(decoded_data)

                contribution_table = calculate_contribution_ratios(
                    eigenvalues, sorted_indices, num_components=2)
                print("Contribution Ratios Table:")
                print(contribution_table)

                print("\n")  # エンコード方法間の区切り
            print("\n")  # 正規化方法間の区切り
        print("\n")  # データセット間の区切り


if __name__ == "__main__":
    main()
