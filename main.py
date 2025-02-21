import pandas as pd
import gradio as gr
import gc
import numpy as np
import re
from catboost import CatBoostClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class DataPreprocessorOptimized:
    """
    Класс для предобработки данных.
    Выполняет:
      - Очистку текстовых признаков
      - Обработку временных и финансовых признаков
      - Обработку пропусков
      - Кодирование категориальных признаков
    """
    def __init__(self):
        # Инициализируем регулярные выражения и энкодеры
        self._init_regex_patterns()
        self._init_encoders()

    def _init_regex_patterns(self):
        """
        Инициализирует регулярные выражения для очистки текстовых признаков,
        таких как 'region' и 'settlement'. Здесь задаются слова, которые необходимо удалить,
        а также шаблоны для классификации населённых пунктов.
        """
        remove_words = [
            'обл', 'область', 'край', 'народная', 'респ', 'г',
            'республика', 'аобл', 'район', 'ао', 'автономный округ',
            'югра', 'якутия', 'кузбасс', 'алания', 'чувашия'
        ]
        self.pattern_remove_words = re.compile(r'\b(?:' + '|'.join(remove_words) + r')\b', re.I)
        self.pattern_non_word = re.compile(r'[^\w\s]', re.I)
        self.pattern_sakhalin = re.compile(r'\bсахалин\b', re.I)
        # Шаблоны для определения типа населённого пункта:
        # 2 – для поселков городского типа, 3 – для поселков/сел, 1 – для городов.
        self.settlement_patterns = {
            2: re.compile(r'\b(пгт|посёлок городского типа|поселок городского типа)\b', re.I),
            3: re.compile(r'\b(п|рп|поселок|посёлок|с|сп|село)\b', re.I),
            1: re.compile(r'\b(г|город)\b', re.I)
        }

    def _init_encoders(self):
        """
        Инициализирует энкодеры для категориальных признаков:
          - onehot_encoder для столбцов с небольшим числом уникальных значений.
          - ordinal_encoder для остальных.
        """
        self.onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        self.ordinal_encoder = OrdinalEncoder()

    def _process_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обрабатывает текстовые признаки:
          - Нормализует столбец 'region': приводит к нижнему регистру, удаляет лишние слова, спецсимволы,
            убирает дублирование слов.
          - Обрабатывает 'settlement': нормализует, создает категорию населённого пункта.
        """
        if 'region' in df.columns:
            df['region'] = (
                df['region']
                .str.lower()
                .str.replace(self.pattern_remove_words, '', regex=True)
                .str.replace(self.pattern_non_word, '', regex=True)
                .str.replace(r'\s+', ' ', regex=True)
                .str.replace(self.pattern_sakhalin, 'сахалинская', regex=True)
                .apply(lambda x: ' '.join(dict.fromkeys(x.split())))
                .str.strip()
            )
        if 'settlement' in df.columns:
            s = df['settlement'].str.lower()
            conditions = [
                s.str.contains(self.settlement_patterns[2], regex=True),
                s.str.contains(self.settlement_patterns[3], regex=True),
                s.str.contains(self.settlement_patterns[1], regex=True)
            ]
            df['settlement_category'] = np.select(conditions, [2, 3, 1], default=4)
        return df

    def _process_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обрабатывает временные признаки:
          - Извлекает базовые признаки из 'created_at': created_month, created_dayofweek.
          - Вычисляет: time_between_loans, time_between_loans_mean, first_loan, last_loan, loan_count,
            time_since_first_loan, time_since_last_loan, loan_frequency.
        """
        # Сортировка по client_id и created_at
        df = df.sort_values(['client_id', 'created_at'])
        # Вычисляем разницу по времени между займами для каждой группы
        df['time_between_loans'] = df.groupby('client_id')['created_at'].diff().dt.days

        # Группируем по всему DataFrame по 'client_id'
        g = df.groupby('client_id')
        # Агрегируем нужные столбцы
        agg_df = g.agg({
            'time_between_loans': 'mean',
            'created_at': ['min', 'max'],
            'loan_id': 'size'
        }).reset_index()
        # Приводим многоуровневый индекс столбцов к плоскому виду
        agg_df.columns = ['client_id', 'time_between_loans_mean', 'first_loan', 'last_loan', 'loan_count']

        # Объединяем агрегированные данные с исходным DataFrame
        df = df.merge(agg_df, on='client_id', how='left')
        df['time_since_first_loan'] = (df['created_at'] - df['first_loan']).dt.days
        df['time_since_last_loan'] = (df['last_loan'] - df['created_at']).dt.days
        df['loan_frequency'] = df['loan_count'] / (((df['last_loan'] - df['first_loan']).dt.days / 365) + 1e-6)
        df.drop(columns=['first_loan', 'last_loan'], inplace=True)
        # Базовые временные признаки
        df['created_month'] = df['created_at'].dt.month
        df['created_dayofweek'] = df['created_at'].dt.dayofweek
        return df

    def _process_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обрабатывает финансовые признаки:
          - Вычисляет коэффициент approved_requested_ratio.
        """
        if 'approved_amount' in df.columns and 'requested_amount' in df.columns:
            df['approved_requested_ratio'] = df['approved_amount'] / (df['requested_amount'] + 1e-6)
        return df

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Заполняет пропуски медианой для всех числовых признаков, кроме идентификаторов.
        """
        numeric_cols = df.select_dtypes(include=np.number).columns.difference(['client_id', 'loan_id'])
        if not numeric_cols.empty:
            medians = df[numeric_cols].median()
            df[numeric_cols] = df[numeric_cols].fillna(medians)
        return df

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Кодирует категориальные признаки:
          - Создает комбинированный признак 'status_client_type'.
          - Для признаков с ≤10 уникальными значениями применяется one-hot кодирование; исходные колонки остаются.
          - Для остальных создается новый столбец с порядковым кодированием (с суффиксом '_ordinal').
        """
        if 'status' in df.columns and 'client_type' in df.columns:
            df['status_client_type'] = df['status'].astype(str) + "_" + df['client_type'].astype(str)
            df['status_client_type'] = df['status_client_type'].astype('category')
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in cat_cols:
            if df[col].nunique() <= 10:
                self.onehot_encoder.fit(df[[col]])
                encoded = self.onehot_encoder.transform(df[[col]])
                new_cols = self.onehot_encoder.get_feature_names_out([col])
                df = pd.concat([df, pd.DataFrame(encoded, columns=new_cols, index=df.index)], axis=1)
            else:
                df[col + '_ordinal'] = self.ordinal_encoder.fit_transform(df[[col]]).astype(np.int32)
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет последовательную предобработку данных.
        Шаги:
          1. Обработка текстовых признаков (region, settlement).
          2. Обработка временных признаков (создание базовых временных признаков и групповых статистик).
          3. Обработка финансовых признаков (расчет коэффициентов и производных признаков).
          4. Обработка пропусков (заполнение медианой).
          5. Кодирование категориальных признаков.
        После каждого шага вызывается сборщик мусора для оптимизации памяти.
        """
        processing_steps = [
            self._process_text_features,
            self._process_time_features,
            self._process_financial_features,
            self._handle_missing_data,
            self._encode_features,
        ]
        for step in processing_steps:
            df = step(df).copy()
            gc.collect()
        return df

# Загрузка модели
catboost_model = CatBoostClassifier()
catboost_model.load_model("catboost_model.cbm")

non_encoded_cat_features = [
    'approved_requested_ratio', 'contact_cases', 'created_dayofweek',
    'created_month', 'first_source', 'gender', 'have_extension',
    'interface', 'loan_count', 'loan_frequency', 'loan_order',
    'payment_frequency', 'region', 'repayment_type', 'settlement',
    'settlement_category', 'source', 'status_client_type',
    'time_between_loans', 'time_between_loans_mean',
    'time_since_first_loan', 'time_since_last_loan', 'type'
]

def predict(file):
    # При загрузке CSV преобразуем колонки 'created_at' и 'start_dt' в даты
    df = pd.read_csv(file.name, parse_dates=["created_at", "start_dt"])
    preprocessor = DataPreprocessorOptimized()
    processed_df = preprocessor.preprocess_data(df)
    processed_df = processed_df.drop(columns=['created_at', 'start_dt'], errors='ignore')
    preds = catboost_model.predict(processed_df[non_encoded_cat_features])
    submission = pd.DataFrame({
        "loan_id": processed_df["loan_id"],
        "churn": preds
    })
    submission_path = "submission_catboost.csv"
    submission.to_csv(submission_path, index=False)
    return submission_path

gr_interface = gr.Interface(
    fn=predict,
    inputs=gr.File(label="Загрузите CSV-файл"),
    outputs=gr.File(label="Скачать предсказания"),
    title="Предсказание оттока клиентов"
)

if __name__ == "__main__":
    gr_interface.launch(server_name="0.0.0.0", server_port=7860)