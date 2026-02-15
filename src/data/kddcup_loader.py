"""
KDD Cup 99 数据集预处理器

支持 KDD Cup 1999 网络入侵检测数据集
适配 Windows 系统

数据集下载: http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
"""
import os
import pandas as pd
import numpy as np
import pickle
from typing import Tuple, Dict, List, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')


class KDDCupPreprocessor:
    """
    KDD Cup 99 数据集预处理器

    功能：
    - 加载 KDD Cup 数据集（支持多种格式）
    - 数据清洗和特征编码
    - 标签编码（二分类/多分类）
    - 数据标准化
    """

    # KDD Cup 99 标准列名（41个特征 + 1个标签）
    COLUMN_NAMES = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label'
    ]

    # 分类特征列
    CATEGORICAL_COLUMNS = ['protocol_type', 'service', 'flag']

    # 数值特征列
    NUMERICAL_COLUMNS = [col for col in COLUMN_NAMES[:-1] if col not in ['protocol_type', 'service', 'flag']]

    # 攻击类型映射到大类
    ATTACK_CATEGORIES = {
        'normal': 'normal',
        # DoS 攻击
        'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
        'smurf': 'dos', 'teardrop': 'dos', 'apache2': 'dos', 'udpstorm': 'dos',
        'processtable': 'dos', 'mailbomb': 'dos',
        # Probe 探测攻击
        'satan': 'probe', 'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
        'mscan': 'probe', 'saint': 'probe',
        # R2L 远程到本地攻击
        'guess_passwd': 'r2l', 'ftp_write': 'r2l', 'imap': 'r2l', 'phf': 'r2l',
        'multihop': 'r2l', 'warezmaster': 'r2l', 'warezclient': 'r2l', 'spy': 'r2l',
        'xlock': 'r2l', 'xsnoop': 'r2l', 'snmpguess': 'r2l', 'snmpgetattack': 'r2l',
        'httptunnel': 'r2l', 'sendmail': 'r2l', 'named': 'r2l', 'worm': 'r2l',
        # U2R 用户到根攻击
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'rootkit': 'u2r', 'perl': 'u2r',
        'sqlattack': 'u2r', 'xterm': 'u2r', 'ps': 'u2r',
    }

    # 特征组定义（用于多源数据分割）
    FEATURE_GROUPS = {
        'basic': [  # 基本连接特征
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent'
        ],
        'content': [  # 内容特征
            'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login'
        ],
        'traffic': [  # 流量特征
            'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate'
        ],
        'host': [  # 主机特征
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
    }

    def __init__(self, config: Dict = None):
        """初始化预处理器"""
        self.config = config or {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.categorical_encoders = {}
        self.feature_names = None
        self.num_classes = None
        self.class_names = None

    def load_data(self, file_path: str, has_header: bool = False) -> pd.DataFrame:
        """
        加载 KDD Cup 数据文件

        Args:
            file_path: 数据文件路径（支持 .csv, .data, .txt）
            has_header: 文件是否包含表头

        Returns:
            DataFrame
        """
        print(f"加载文件: {file_path}")

        # 尝试不同的编码方式
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk']

        for encoding in encodings:
            try:
                if has_header:
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                else:
                    df = pd.read_csv(file_path, names=self.COLUMN_NAMES,
                                    encoding=encoding, low_memory=False)
                print(f"  - 成功使用 {encoding} 编码加载")
                print(f"  - 数据形状: {df.shape}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"  - 加载错误 ({encoding}): {e}")
                continue

        raise ValueError(f"无法加载文件: {file_path}")

    def clean_labels(self, labels: pd.Series) -> pd.Series:
        """清理标签（去除末尾的点号等）"""
        return labels.str.strip().str.rstrip('.')

    def encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        编码分类特征

        Args:
            df: 数据DataFrame
            fit: 是否fit编码器

        Returns:
            编码后的DataFrame
        """
        df = df.copy()

        for col in self.CATEGORICAL_COLUMNS:
            if col in df.columns:
                if fit:
                    self.categorical_encoders[col] = LabelEncoder()
                    df[col] = self.categorical_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # 处理未见过的类别
                    known_classes = set(self.categorical_encoders[col].classes_)
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in known_classes else 'unknown'
                    )
                    if 'unknown' not in known_classes:
                        self.categorical_encoders[col].classes_ = np.append(
                            self.categorical_encoders[col].classes_, 'unknown'
                        )
                    df[col] = self.categorical_encoders[col].transform(df[col])

        return df

    def encode_labels(
        self,
        labels: pd.Series,
        binary: bool = False,
        use_categories: bool = True,
        fit: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        编码标签

        Args:
            labels: 标签Series
            binary: 是否二分类（正常 vs 攻击）
            use_categories: 是否使用攻击大类（5分类）
            fit: 是否fit encoder

        Returns:
            编码后的标签和类别名称
        """
        labels = self.clean_labels(labels)

        if binary:
            # 二分类：normal=0, attack=1
            encoded = (labels != 'normal').astype(int)
            class_names = ['Normal', 'Attack']
        elif use_categories:
            # 5分类：normal, dos, probe, r2l, u2r
            labels_mapped = labels.map(
                lambda x: self.ATTACK_CATEGORIES.get(x.lower(), 'unknown')
            )
            if fit:
                self.label_encoder.fit(labels_mapped)
            encoded = self.label_encoder.transform(labels_mapped)
            class_names = list(self.label_encoder.classes_)
        else:
            # 多分类：保留原始标签
            if fit:
                encoded = self.label_encoder.fit_transform(labels)
                class_names = list(self.label_encoder.classes_)
            else:
                encoded = self.label_encoder.transform(labels)
                class_names = list(self.label_encoder.classes_)

        self.num_classes = len(class_names)
        self.class_names = class_names

        return encoded, class_names

    def normalize_features(
        self,
        X: np.ndarray,
        method: str = 'standard',
        fit: bool = True
    ) -> np.ndarray:
        """特征标准化"""
        if method == 'minmax':
            if not hasattr(self, 'minmax_scaler'):
                self.minmax_scaler = MinMaxScaler()
            scaler = self.minmax_scaler
        else:
            scaler = self.scaler

        if fit:
            return scaler.fit_transform(X)
        return scaler.transform(X)

    def preprocess(
        self,
        data_path: str,
        binary_classification: bool = False,
        use_categories: bool = True,
        normalize: bool = True,
        sample_size: Optional[int] = None,
        has_header: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        完整的预处理流程

        Args:
            data_path: 数据文件路径
            binary_classification: 是否二分类
            use_categories: 是否使用攻击大类
            normalize: 是否标准化
            sample_size: 采样数量（None表示使用全部数据）
            has_header: 文件是否包含表头

        Returns:
            包含特征和标签的字典
        """
        print("=" * 50)
        print("KDD Cup 99 数据预处理")
        print("=" * 50)

        # 1. 加载数据
        df = self.load_data(data_path, has_header=has_header)

        # 2. 采样（如果指定）
        if sample_size and len(df) > sample_size:
            print(f"\n采样 {sample_size} 条数据...")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        # 3. 清理数据
        print("\n=== 数据清洗 ===")
        original_shape = df.shape

        # 删除重复行
        df = df.drop_duplicates()
        print(f"删除重复行: {original_shape[0] - df.shape[0]} 行")

        # 处理缺失值
        df = df.dropna()
        print(f"最终数据形状: {df.shape}")

        # 4. 分离特征和标签
        label_col = 'label' if 'label' in df.columns else df.columns[-1]
        feature_cols = [col for col in df.columns if col != label_col]

        # 5. 编码分类特征
        print("\n编码分类特征...")
        df = self.encode_categorical(df)

        # 6. 编码标签
        y, class_names = self.encode_labels(
            df[label_col],
            binary=binary_classification,
            use_categories=use_categories
        )

        print(f"\n类别分布:")
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  {class_names[u]}: {c} ({c/len(y)*100:.2f}%)")

        # 7. 提取特征
        X = df[feature_cols].values.astype(np.float32)
        self.feature_names = feature_cols

        # 8. 标准化
        if normalize:
            X = self.normalize_features(X)
            print("\n已完成特征标准化")

        result = {
            'X': X,
            'y': y,
            'feature_names': feature_cols,
            'class_names': class_names,
            'num_features': X.shape[1],
            'num_classes': len(class_names)
        }

        print("\n预处理完成!")
        print(f"特征维度: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        print(f"类别数量: {len(class_names)}")

        return result

    def save_preprocessed(self, data: Dict, path: str):
        """保存预处理后的数据"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"预处理数据已保存到: {path}")

    @staticmethod
    def load_preprocessed(path: str) -> Dict:
        """加载预处理后的数据"""
        with open(path, 'rb') as f:
            return pickle.load(f)


class KDDMultiSourceSplitter:
    """
    KDD Cup 多源数据分割器

    将特征分割为多个数据源用于融合实验
    """

    # 特征组定义
    FEATURE_GROUPS = KDDCupPreprocessor.FEATURE_GROUPS

    def __init__(
        self,
        source1_groups: List[str] = None,
        source2_groups: List[str] = None
    ):
        """
        初始化分割器

        Args:
            source1_groups: 第一个数据源包含的特征组
            source2_groups: 第二个数据源包含的特征组
        """
        # 默认分割：基本+内容 vs 流量+主机
        self.source1_groups = source1_groups or ['basic', 'content']
        self.source2_groups = source2_groups or ['traffic', 'host']

    def get_feature_indices(
        self,
        feature_names: List[str],
        groups: List[str]
    ) -> List[int]:
        """获取指定特征组的特征索引"""
        target_features = []
        for group in groups:
            target_features.extend(self.FEATURE_GROUPS.get(group, []))

        indices = []
        for i, name in enumerate(feature_names):
            if name in target_features:
                indices.append(i)
        return indices

    def split(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        分割特征为两个数据源

        Args:
            X: 原始特征数组
            feature_names: 特征名称列表

        Returns:
            (source1_features, source2_features, source1_names, source2_names)
        """
        idx1 = self.get_feature_indices(feature_names, self.source1_groups)
        idx2 = self.get_feature_indices(feature_names, self.source2_groups)

        # 确保至少有一些特征
        if not idx1:
            idx1 = list(range(len(feature_names) // 2))
        if not idx2:
            idx2 = list(range(len(feature_names) // 2, len(feature_names)))

        X1 = X[:, idx1]
        X2 = X[:, idx2]

        names1 = [feature_names[i] for i in idx1]
        names2 = [feature_names[i] for i in idx2]

        print(f"\n数据源分割:")
        print(f"  Source 1 ({'+'.join(self.source1_groups)}): {X1.shape[1]} 个特征")
        print(f"  Source 2 ({'+'.join(self.source2_groups)}): {X2.shape[1]} 个特征")

        return X1, X2, names1, names2


def download_kddcup_sample():
    """
    Download KDD Cup 99 10% subset to data/raw.

    Returns:
        Local CSV path when successful, otherwise None.
    """
    import gzip
    import shutil
    import urllib.request
    from urllib.error import HTTPError, URLError

    urls = [
        "https://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
        "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
    ]

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(output_dir, exist_ok=True)

    gz_path = os.path.join(output_dir, "kddcup.data_10_percent.gz")
    csv_path = os.path.join(output_dir, "kddcup_10percent.csv")

    if os.path.exists(csv_path):
        print(f"Data file already exists: {csv_path}")
        return csv_path

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python-urllib"
    }

    for url in urls:
        print(f"Trying to download KDD Cup 99 from: {url}")
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response, open(gz_path, 'wb') as f_out:
                shutil.copyfileobj(response, f_out)

            print("Download completed, extracting...")
            with gzip.open(gz_path, 'rb') as f_in, open(csv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

            os.remove(gz_path)
            print(f"Data saved to: {csv_path}")
            return csv_path
        except (HTTPError, URLError, OSError) as e:
            print(f"Download failed from {url}: {e}")
            if os.path.exists(gz_path):
                try:
                    os.remove(gz_path)
                except OSError:
                    pass

    print("Falling back to sklearn.datasets.fetch_kddcup99...")
    try:
        import numpy as np
        import pandas as pd
        from sklearn.datasets import fetch_kddcup99

        bunch = fetch_kddcup99(percent10=True, subset=None, as_frame=False, download_if_missing=True)
        X = bunch.data
        y = bunch.target

        if getattr(X, 'dtype', None) == object:
            X = np.vectorize(
                lambda v: v.decode('utf-8', errors='ignore') if isinstance(v, (bytes, bytearray)) else v,
                otypes=[object],
            )(X)
        if getattr(y, 'dtype', None) == object:
            y = np.vectorize(
                lambda v: v.decode('utf-8', errors='ignore') if isinstance(v, (bytes, bytearray)) else v,
                otypes=[object],
            )(y)

        df = pd.DataFrame(X, columns=KDDCupPreprocessor.COLUMN_NAMES[:-1])
        df[KDDCupPreprocessor.COLUMN_NAMES[-1]] = y
        df.to_csv(csv_path, index=False, header=False)

        print(f"Data downloaded via sklearn and saved to: {csv_path}")
        return csv_path
    except Exception as e:
        print(f"Failed to download KDD Cup 99 data automatically: {e}")
        print("Please download manually and place it under data/raw/")
        return None
