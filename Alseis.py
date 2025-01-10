import random
import cirq
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, load_diabetes
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
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
        if hasattr(data, 'target'):
            labels = data.target
        else:
            labels = None
    else:
        # data がタプルだった場合
        features = data[0]
        labels = data[1] if len(data) > 1 else None

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown method: {method}")

    scaled_features = scaler.fit_transform(features)
    return scaled_features, labels


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


def create_qpe_circuit(qubits, unitary, ancilla_qubits, num_ancilla):
    """
    QPE回路を作成します。
    qubits: エンコードされたデータ用の量子ビット
    unitary: 固有値を推定するユニタリ演算子
    ancilla_qubits: QPE用の補助量子ビット
    num_ancilla: QPEに使用する補助量子ビットの数
    """
    circuit = cirq.Circuit()
    # 初期化（補助量子ビットを|+>状態にする）
    circuit.append([cirq.H(q) for q in ancilla_qubits])

    # ユニタリ演算子を補助量子ビットに制御して適用
    for i in range(num_ancilla):
        exponent = 2**i
        controlled_unitary = cirq.ControlledGate(unitary ** exponent)
        circuit.append(controlled_unitary.on(ancilla_qubits[i], qubits[0]))

    # 逆フーリエ変換
    circuit.append(cirq.inverse(cirq.qft(*ancilla_qubits)))

    # 測定
    circuit.append([cirq.measure(q, key=f'm{idx}')
                   for idx, q in enumerate(ancilla_qubits)])

    return circuit


def get_unitary_from_density_matrix(density_matrix):
    """
    密度行列からユニタリ演算子を構築します。
    簡易的な例として、密度行列の固有値に基づく回転ゲートを使用します。
    """
    eigenvalues, eigenvectors = eigh(density_matrix)
    # 最も大きな固有値に対応するユニタリ演算子を選択
    principal_eigenvalue = eigenvalues[-1]
    principal_eigenvector = eigenvectors[:, -1]

    # 単一量子ビットの回転ゲートを例として使用
    theta = principal_eigenvalue * 2 * np.pi
    unitary = cirq.ry(theta)

    return unitary, principal_eigenvector


def perform_qpca_qpe(circuits, num_iterations, simulator, config):
    """
    qPCAをQPEを用いて実施し、エンコードされた量子データに対して次元削減を行います。
    """
    config_print_eigenvalues = config.get('print_eigenvalues', False)
    config_print_eigenvectors = config.get('print_eigenvectors', False)

    density_matrices = []
    for circuit in circuits:
        result = simulator.simulate(circuit)
        state = result.final_state_vector
        density_matrix = np.outer(state, np.conj(state))
        density_matrices.append(density_matrix)

    # 全ての密度行列を平均して最終的な密度行列を作成
    avg_density_matrix = np.mean(density_matrices, axis=0)

    # ユニタリ演算子と固有ベクトルを取得
    unitary, principal_eigenvector = get_unitary_from_density_matrix(
        avg_density_matrix)

    # 補助量子ビットの数（精度に影響）
    num_ancilla = num_iterations  # 例として主成分数を使用

    # 補助量子ビットを定義
    ancilla_qubits = [cirq.LineQubit(i) for i in range(num_ancilla)]
    data_qubit = cirq.LineQubit(num_ancilla)

    # QPE回路の作成
    qpe_circuit = create_qpe_circuit(
        [data_qubit], unitary, ancilla_qubits, num_ancilla)

    if config.get('print_circuit', False):
        print("Quantum Phase Estimation Circuit:")
        print(qpe_circuit)

    # シミュレーション
    result = simulator.run(qpe_circuit, repetitions=1)

    # 測定結果の解析
    eigenvalue_bits = [result.measurements[f'm{
        idx}'][0][0] for idx in range(num_ancilla)]
    eigenvalue = 0
    for bit in eigenvalue_bits:
        eigenvalue = (eigenvalue << 1) | bit
    estimated_eigenvalue = eigenvalue / (2**num_ancilla)

    if config_print_eigenvalues:
        print("Estimated Eigenvalue from QPE:")
        print(estimated_eigenvalue)

    # 固有ベクトルの取得（クラシカルに取得）
    eigenvalues, eigenvectors = eigh(avg_density_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_eigenvectors = eigenvectors[:, sorted_indices[:num_iterations]]
    top_eigenvalues = eigenvalues[sorted_indices[:num_iterations]]

    if config_print_eigenvectors:
        print("Top Eigenvectors after qPCA:")
        print(top_eigenvectors)

    return top_eigenvectors, top_eigenvalues, sorted_indices


def perform_classical_pca(scaled_data, num_components=2, config=None):
    """
    古典的なPCAを実施し、次元削減を行います。
    """
    pca = PCA(n_components=num_components)
    transformed_data = pca.fit_transform(scaled_data)
    eigenvalues = pca.explained_variance_
    components = pca.components_
    contribution_ratios = pca.explained_variance_ratio_

    if config and config.get('print_classical_pca', False):
        print("Classical PCA Eigenvalues:")
        print(eigenvalues)
        print("Classical PCA Components:")
        print(components)
        print("Classical PCA Contribution Ratios:")
        print(contribution_ratios)

    return transformed_data, eigenvalues, components, contribution_ratios


def decode_quantum_data(top_eigenvectors, scaled_data):
    """
    qPCAで次元削減した量子データを古典データに低次元空間に射影します
    """
    # 実数部分のみを使用
    top_eigenvectors_reduced = top_eigenvectors[:scaled_data.shape[1], :].real
    reduced_data = np.dot(scaled_data, top_eigenvectors_reduced)
    return reduced_data


def calculate_contribution_ratios(eigenvalues, num_components):
    """
    寄与率を計算し、各主成分の寄与率の表を作成します。
    """
    total_variance = np.sum(eigenvalues)
    num_available = len(eigenvalues)
    num_selected = min(num_components, num_available)  # 利用可能な固有値数に制限

    if num_selected < num_components:
        print(f"Warning: Requested num_components={num_components} exceeds available eigenvalues={
              num_available}. Adjusting to {num_selected}.")

    # sorted_indicesを使用せず、先頭のnum_selectedを取得
    selected_eigenvalues = eigenvalues[:num_selected]
    contribution_ratios = selected_eigenvalues / total_variance

    contribution_table = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(num_selected)],
        'Eigenvalue': selected_eigenvalues,
        'Contribution Ratio': contribution_ratios
    })

    return contribution_table


def plot_original_data(scaled_data, labels, dataset_name, config):
    """
    元のデータを2次元でプロットします。データが2次元より高次元の場合はPCAで2次元に次元削減します。
    """
    plot_config = config.get('plotting', {})
    plot_original = plot_config.get('plot_original_data', False)
    save_fig = plot_config.get('save_figures', False)
    fig_path = plot_config.get('original_data_fig_path', f"{
                               dataset_name}_original_data.png")

    if not plot_original:
        return

    if scaled_data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(scaled_data)
        title = f"{dataset_name} Original Data (PCA Reduced to 2D)"
    else:
        data_2d = scaled_data
        title = f"{dataset_name} Original Data"

    # 実数部分のみを使用
    data_2d = data_2d.real

    plt.figure(figsize=(8, 6))
    if labels is not None:
        unique_labels = np.unique(labels)
        palette = sns.color_palette("hsv", len(unique_labels))
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1],
                        hue=labels, palette=palette, legend='full')
    else:
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1])
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_path)
        print(f"Original data plot saved to {fig_path}")
    else:
        plt.show()
    plt.close()


def plot_qpca_results(decoded_data, labels, dataset_name, config):
    """
    qPCA後の低次元データをプロットします。
    """
    plot_config = config.get('plotting', {})
    plot_qpca = plot_config.get('plot_qpca_results', False)
    save_fig = plot_config.get('save_figures', False)
    fig_path = plot_config.get('qpca_results_fig_path', f"{
                               dataset_name}_qpca_results.png")

    if not plot_qpca:
        return

    if decoded_data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(decoded_data)
        title = f"{dataset_name} qPCA Results (PCA Reduced to 2D)"
    else:
        data_2d = decoded_data
        title = f"{dataset_name} qPCA Results"

    # 実数部分のみを使用
    data_2d = data_2d.real

    plt.figure(figsize=(8, 6))
    if labels is not None:
        unique_labels = np.unique(labels)
        palette = sns.color_palette("hsv", len(unique_labels))
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1],
                        hue=labels, palette=palette, legend='full')
    else:
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1])
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_path)
        print(f"qPCA results plot saved to {fig_path}")
    else:
        plt.show()
    plt.close()


def plot_classical_pca_results(classical_pca_data, labels, dataset_name, config):
    """
    古典PCA後の低次元データをプロットします。
    """
    plot_config = config.get('plotting', {})
    plot_classical_pca = plot_config.get('plot_classical_pca_results', False)
    save_fig = plot_config.get('save_figures', False)
    fig_path = plot_config.get('classical_pca_results_fig_path', f"{
                               dataset_name}_classical_pca_results.png")

    if not plot_classical_pca:
        return

    if classical_pca_data.shape[1] > 2:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(classical_pca_data)
        title = f"{dataset_name} Classical PCA Results (PCA Reduced to 2D)"
    else:
        data_2d = classical_pca_data
        title = f"{dataset_name} Classical PCA Results"

    # 実数部分のみを使用
    data_2d = data_2d.real

    plt.figure(figsize=(8, 6))
    if labels is not None:
        unique_labels = np.unique(labels)
        palette = sns.color_palette("hsv", len(unique_labels))
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1],
                        hue=labels, palette=palette, legend='full')
    else:
        sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1])
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_path)
        print(f"Classical PCA results plot saved to {fig_path}")
    else:
        plt.show()
    plt.close()


def plot_eigenvalues(eigenvalues, dataset_name, config, method='qPCA'):
    """
    固有値をプロットし、寄与率を視覚化します。
    method: 'qPCA' または 'Classical PCA'
    """
    plot_config = config.get('plotting', {})
    plot_eigen = plot_config.get('plot_eigenvalues', False)
    save_fig = plot_config.get('save_figures', False)
    fig_path = plot_config.get('eigenvalues_fig_path', f"{
                               dataset_name}_eigenvalues_{method}.png")

    if not plot_eigen:
        return

    num_components = len(eigenvalues)
    sorted_eigenvalues = eigenvalues  # 既にソート済み

    plt.figure(figsize=(8, 6))

    sns.barplot(x=np.arange(1, num_components + 1),
                y=sorted_eigenvalues, color="skyblue")

    plt.title(f"{dataset_name} {method} Eigenvalues (Sorted)")
    plt.xlabel("Component")
    plt.ylabel("Eigenvalue")
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_path)
        print(f"{method} Eigenvalues plot saved to {fig_path}")
    else:
        plt.show()
    plt.close()


def plot_pca_comparison(qpca_eigenvalues,
                        classical_eigenvalues,
                        qpca_contribution,
                        classical_contribution,
                        dataset_name, config):
    """
    qPCAと古典PCAの固有値および寄与率を比較してプロットします。
    """
    plot_config = config.get('plotting', {})
    plot_pca_comparison_flag = plot_config.get('plot_pca_comparison', False)
    save_fig = plot_config.get('save_figures', False)
    fig_path_eigen = plot_config.get('pca_comparison_eigenvalues_fig_path', f"{
                                     dataset_name}_pca_comparison_eigenvalues.png")
    fig_path_contribution = plot_config.get('pca_comparison_contribution_fig_path', f"{
                                            dataset_name}_pca_comparison_contribution.png")

    if not plot_pca_comparison_flag:
        return

    num_components = len(qpca_eigenvalues)  # qPCAと古典PCAのコンポーネント数が一致している前提

    # 固有値の比較: 上位num_componentsのみ
    sorted_qpca_eigenvalues = qpca_eigenvalues
    sorted_classical_eigenvalues = classical_eigenvalues

    plt.figure(figsize=(10, 6))
    width = 0.35  # バーの幅
    indices = np.arange(num_components)

    plt.bar(indices - width/2, sorted_qpca_eigenvalues,
            width=width, label='qPCA', color='skyblue')
    plt.bar(indices + width/2, sorted_classical_eigenvalues,
            width=width, label='Classical PCA', color='salmon')

    plt.title(f"{dataset_name} PCA Comparison of Eigenvalues")
    plt.xlabel("Component")
    plt.ylabel("Eigenvalue")
    plt.xticks(indices, [f'PC{i+1}' for i in range(num_components)])
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_path_eigen)
        print(f"PCA comparison Eigenvalues plot saved to {fig_path_eigen}")
    else:
        plt.show()
    plt.close()

    # 寄与率の比較: 上位num_componentsのみ
    sorted_qpca_contribution = qpca_contribution
    sorted_classical_contribution = classical_contribution

    plt.figure(figsize=(10, 6))
    width = 0.35  # バーの幅
    indices = np.arange(num_components)

    plt.bar(indices - width/2, sorted_qpca_contribution,
            width=width, label='qPCA', color='skyblue')
    plt.bar(indices + width/2, sorted_classical_contribution,
            width=width, label='Classical PCA', color='salmon')

    plt.title(f"{dataset_name} PCA Comparison of Contribution Ratios")
    plt.xlabel("Component")
    plt.ylabel("Contribution Ratio")
    plt.xticks(indices, [f'PC{i+1}' for i in range(num_components)])
    plt.legend()
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_path_contribution)
        print(f"PCA comparison Contribution Ratios plot saved to {
              fig_path_contribution}")
    else:
        plt.show()
    plt.close()


def main():
    # 設定ファイルのパスを指定
    config = load_config(CONFIG_PATH)

    # シードの設定
    seed = set_random_seed(config)

    dataset_names = config.get(
        'dataset_name', 'iris').split(',')   # 複数データセット対応
    normalization_methods = config.get(
        'normalization_method', 'standard').split(',')
    encoding_methods = config.get('encoding_method', 'amplitude').split(',')
    config_print_circuit = config.get('print_circuit', False)
    config_print_top_eigenvectors = config.get('print_eigenvectors', False)
    config_print_classical_data = config.get('print_classical_data', False)

    # Cirqのシミュレーターにシードを渡す（固定シードが必要な場合）
    if seed is not None:
        simulator = cirq.Simulator(seed=seed)
    else:
        simulator = cirq.Simulator()

    desired_num_components = config.get('num_components', 2)  # 追加

    for dname in dataset_names:
        # データセットごとに処理
        print(f"=== Dataset: {dname} ===")
        for norm_method in normalization_methods:
            print(f"Testing with {norm_method} normalization:")
            scaled_data, labels = load_and_scale_dataset(
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

                # qPCAをQPEを用いて実行
                top_eigenvectors, top_eigenvalues, sorted_indices = perform_qpca_qpe(
                    circuits, num_iterations=desired_num_components, simulator=simulator, config=config)
                if config_print_top_eigenvectors:
                    print("Top Eigenvalues after qPCA (QPE):")
                    print(top_eigenvalues)
                    print("Top Eigenvectors after qPCA:")
                    print(top_eigenvectors)

                # デコードされたデータを取得
                decoded_data = decode_quantum_data(
                    top_eigenvectors, scaled_data)
                if config_print_classical_data:
                    print("Decoded Classical Data:")
                    print(decoded_data)

                # 固有値の数に応じてnum_componentsを調整
                actual_num_components = min(
                    desired_num_components, len(top_eigenvalues))

                contribution_table = calculate_contribution_ratios(
                    top_eigenvalues, num_components=actual_num_components)

                print("qPCA Contribution Ratios Table:")
                print(contribution_table)

                # 古典PCAの実施
                perform_classical_pca_flag = config.get(
                    'perform_classical_pca', False)
                if perform_classical_pca_flag:
                    classical_pca_data, classical_eigenvalues, classical_components, classical_contribution_ratios = perform_classical_pca(
                        scaled_data, num_components=desired_num_components, config=config)
                    # 古典PCAの固有値数を調整
                    actual_classical_num_components = min(
                        desired_num_components, len(classical_eigenvalues))
                    classical_contribution_table = pd.DataFrame({
                        'Component': [f'PC{i+1}' for i in range(actual_classical_num_components)],
                        'Eigenvalue': classical_eigenvalues[:actual_classical_num_components],
                        'Contribution Ratio': classical_contribution_ratios[:actual_classical_num_components]
                    })
                    print("Classical PCA Contribution Ratios Table:")
                    print(classical_contribution_table)

                # プロットの生成
                plotting_methods = config.get('plotting', {})
                if plotting_methods.get('plot_original_data', False):
                    plot_original_data(scaled_data, labels,
                                       dname.strip(), config)
                if plotting_methods.get('plot_qpca_results', False):
                    plot_qpca_results(decoded_data, labels,
                                      dname.strip(), config)
                if plotting_methods.get('plot_classical_pca_results', False) and perform_classical_pca_flag:
                    plot_classical_pca_results(
                        classical_pca_data, labels, dname.strip(), config)
                if plotting_methods.get('plot_eigenvalues', False):
                    plot_eigenvalues(
                        top_eigenvalues[:actual_num_components], dname.strip(), config, method='qPCA')
                    if perform_classical_pca_flag:
                        sorted_classical_eigenvalues = np.sort(
                            classical_eigenvalues)[::-1]
                        plot_eigenvalues(
                            sorted_classical_eigenvalues[:actual_num_components],
                            dname.strip(), config, method='Classical PCA')
                if plotting_methods.get('plot_pca_comparison', False) and perform_classical_pca_flag:
                    # 固有値と寄与率を上位num_componentsに限定して比較プロットを作成
                    sorted_classical_eigenvalues = np.sort(
                        classical_eigenvalues)[::-1]
                    plot_pca_comparison(
                        top_eigenvalues[:actual_num_components],
                        sorted_classical_eigenvalues[:actual_num_components],
                        contribution_table['Contribution Ratio'].values[:actual_num_components],
                        classical_contribution_ratios[:actual_num_components],
                        dname.strip(), config)

                print("\n")  # エンコード方法間の区切り
            print("\n")  # 正規化方法間の区切り
        print("\n")  # データセット間の区切り


if __name__ == "__main__":
    main()
