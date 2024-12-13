import cirq
import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.linalg import eigh

CONFIG_PTAH = 'config.yaml'


def load_config(config_path):
    """
    YAML形式の設定ファイルを読み込む関数。
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_and_scale_iris(method='standard'):
    """
    アイリスデータセットを読み込み、指定された方法で標準化または正規化して返します。
    method: 'standard', 'minmax', 'robust' のいずれかを指定
    """
    data = load_iris()
    iris_df = pd.DataFrame(data.data, columns=data.feature_names)

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown method: {method}")

    scaled_features = scaler.fit_transform(iris_df)
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


def perform_qpca(qubits, circuits, num_iterations):
    """
    qPCAを実施し、エンコードされた量子データに対して次元削減を行います。
    """
    config = load_config(CONFIG_PTAH)
    config_print_eigenvalues = config.get('print_eigenvalues', False)
    config_print_eigenvectors = config.get('print_eigenvectors', False)

    simulator = cirq.Simulator()
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
    config = load_config(CONFIG_PTAH)
    normalization_method = config.get(
        'normalization_method', 'standard').split(',')
    encoding_method = config.get('encoding_method', 'amplitude').split(',')
    config_print_circuit = config.get('print_circuit', False)
    config_print_top_eingenvectors = config.get(
        'print_top_eingenvectors', False)
    config_print_clasical_data = config.get('print_clasical_data', False)

    # アイリスデータセットの読み込みと標準化
    all_scaled_data = {}
    for method in normalization_method:
        print(f"Testing with {method} normalization:")
        scaled_data = load_and_scale_iris(method)
        all_scaled_data[method] = scaled_data

        # 量子ビットの準備
        num_qubits = scaled_data.shape[1]
        qubits = [cirq.LineQubit(i) for i in range(num_qubits)]

        # 各サンプルを量子エンコードし、回路を作成
        all_circuits = {}
        for norm_method, scaled_data in all_scaled_data.items():
            for method in encoding_method:
                print(
                    f"Testing with {method} encoding (Normalization: {norm_method}):")
                circuits = []
                for data in scaled_data:
                    circuits.append(
                        create_quantum_encoding_circuit(qubits, data, method))
                    # 複数のサンプルに対してより多くのエンコードを行うためにサンプル数を増やす
                    for _ in range(5):  # 各サンプルに対して追加でエンコードを作成（例として5回）
                        # ノイズの標準偏差を小さくして調整
                        noisy_data = data + \
                            np.random.normal(0, 0.005, data.shape)
                        noisy_data = np.clip(noisy_data, -1.0, 1.0)
                        noisy_data = np.nan_to_num(
                            noisy_data)  # NaNが発生した場合にゼロに置き換える
                        circuits.append(create_quantum_encoding_circuit(
                            qubits, noisy_data, method))
                all_circuits[method] = circuits

                if config_print_circuit:
                    # 作成した各回路を表示
                    for i, circuit in enumerate(circuits):
                        print(f"Quantum Encoding Circuit for sample {i}:")
                        print(circuit)

                # qPCAの実行
                top_eigenvectors, eigenvalues, sorted_indices = perform_qpca(
                    qubits, all_circuits[method], num_iterations=2)
                if config_print_top_eingenvectors:
                    print("Top Eigenvectors after qPCA:")
                    print(top_eigenvectors)

                # qPCAで次元削減したデータを低次元空間に射影して古典データに戻す
                decoded_data = decode_quantum_data(
                    top_eigenvectors, scaled_data)
                if config_print_clasical_data:
                    print("Decoded Classical Data:")
                    print(decoded_data)

                # 寄与率の計算と表の作成
                contribution_table = calculate_contribution_ratios(
                    eigenvalues, sorted_indices, num_components=2)
                print("Contribution Ratios Table:")
                print(contribution_table)


if __name__ == "__main__":
    main()
