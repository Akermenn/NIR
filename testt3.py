import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import RobustScaler
from minisom import MiniSom
from fcmeans import FCM


# 1. Генерация данных, приближенных к реальным пробам руды
def generate_ore_data(n_samples=365):
    """Генерация данных с явными кластерами, но с шумом"""
    np.random.seed(42)
    # Основные кластеры (соответствуют сортам руды)
    cluster1 = np.random.normal(loc=[60, 25, 45], scale=[5, 3, 8],
                                size=(n_samples // 3, 3))  # Высокое Fe, легкоизмельчаемый
    cluster2 = np.random.normal(loc=[45, 35, 65], scale=[7, 4, 12], size=(n_samples // 3, 3))  # Средний сорт
    cluster3 = np.random.normal(loc=[70, 40, 30], scale=[6, 5, 10], size=(n_samples // 3, 3))  # Трудноизмельчаемый

    # Добавляем 10% выбросов для реалистичности
    outliers = np.random.uniform(low=[30, 15, 20], high=[80, 50, 80], size=(n_samples // 10, 3))
    return np.vstack((cluster1, cluster2, cluster3, outliers))


data = generate_ore_data()

# 2. Масштабирование (устойчивое к выбросам)
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data)

# 3. Параметры из отчета
window_size = 250  # Соответствует оригинальному исследованию
n_clusters = 3  # Три сорта руды
results = []


# 4. Настройка алгоритмов для соответствия отчету
def kmeans_cluster(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels, kmeans.cluster_centers_


def fuzzy_cmeans(data, n_clusters):
    fcm = FCM(n_clusters=n_clusters, m=1.7)  # m=1.7 для менее "размытых" кластеров
    fcm.fit(data)
    labels = fcm.predict(data)
    return labels, fcm.centers


def kohonen_cluster(data, n_clusters):
    som = MiniSom(1, n_clusters, data.shape[1], sigma=0.2, learning_rate=0.5, random_seed=42)
    som.train_random(data, 150)
    labels = np.array([som.winner(x)[1] for x in data])
    centers = np.array([som.get_weights()[0, i] for i in range(n_clusters)])
    return labels, centers


# 5. Скользящее окно с расчетом метрик
for i in range(0, len(data) - window_size + 1, 10):  # Шаг=10 для ускорения
    window_data = scaled_data[i:i + window_size]

    # K-means
    kmeans_labels, _ = kmeans_cluster(window_data, n_clusters)
    kmeans_ks = silhouette_score(window_data, kmeans_labels)
    kmeans_ch = calinski_harabasz_score(window_data, kmeans_labels)
    kmeans_db = davies_bouldin_score(window_data, kmeans_labels)

    # Fuzzy C-means
    fcm_labels, _ = fuzzy_cmeans(window_data, n_clusters)
    fcm_ks = silhouette_score(window_data, fcm_labels)
    fcm_ch = calinski_harabasz_score(window_data, fcm_labels)
    fcm_db = davies_bouldin_score(window_data, fcm_labels)

    # Kohonen
    kohonen_labels, _ = kohonen_cluster(window_data, n_clusters)
    kohonen_ks = silhouette_score(window_data, kohonen_labels)
    kohonen_ch = calinski_harabasz_score(window_data, kohonen_labels)
    kohonen_db = davies_bouldin_score(window_data, kohonen_labels)

    results.append({
        'window_start': i,
        'kmeans_ks': kmeans_ks,
        'kmeans_ch': kmeans_ch,
        'kmeans_db': kmeans_db,
        'fcm_ks': fcm_ks,
        'fcm_ch': fcm_ch,
        'fcm_db': fcm_db,
        'kohonen_ks': kohonen_ks,
        'kohonen_ch': kohonen_ch,
        'kohonen_db': kohonen_db
    })

# 6. Анализ результатов
results_df = pd.DataFrame(results)

# Графики как в отчете
plt.figure(figsize=(12, 8))
plt.suptitle('Сравнение методов кластеризации для руды', y=1.02)

# Коэффициент силуэта
plt.subplot(3, 1, 1)
plt.plot(results_df['window_start'], results_df['kmeans_ks'], 'b-', label='K-means')
plt.plot(results_df['window_start'], results_df['fcm_ks'], 'g--', label='Fuzzy C-means')
plt.plot(results_df['window_start'], results_df['kohonen_ks'], 'r:', label='Kohonen')
plt.ylabel('KS')
plt.legend(loc='upper right')

# Индекс Калински-Харабаса
plt.subplot(3, 1, 2)
plt.plot(results_df['window_start'], results_df['kmeans_ch'], 'b-')
plt.plot(results_df['window_start'], results_df['fcm_ch'], 'g--')
plt.plot(results_df['window_start'], results_df['kohonen_ch'], 'r:')
plt.ylabel('CH')

# Индекс Дэвиса-Болдуина
plt.subplot(3, 1, 3)
plt.plot(results_df['window_start'], results_df['kmeans_db'], 'b-')
plt.plot(results_df['window_start'], results_df['fcm_db'], 'g--')
plt.plot(results_df['window_start'], results_df['kohonen_db'], 'r:')
plt.ylabel('DB')
plt.xlabel('Номер окна')

plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Вывод средних значений (как в отчете)
print("\nСредние значения метрик:")
print("K-means:")
print(f"  KS: {results_df['kmeans_ks'].mean():.4f} (чем ближе к 1, тем лучше)")
print(f"  CH: {results_df['kmeans_ch'].mean():.1f} (чем выше, тем лучше)")
print(f"  DB: {results_df['kmeans_db'].mean():.4f} (чем ближе к 0, тем лучше)")

print("\nFuzzy C-means:")
print(f"  KS: {results_df['fcm_ks'].mean():.4f}")
print(f"  CH: {results_df['fcm_ch'].mean():.1f}")
print(f"  DB: {results_df['fcm_db'].mean():.4f}")

print("\nKohonen:")
print(f"  KS: {results_df['kohonen_ks'].mean():.4f}")
print(f"  CH: {results_df['kohonen_ch'].mean():.1f}")
print(f"  DB: {results_df['kohonen_db'].mean():.4f}")

# 8. Сохранение результатов в CSV
results_df.to_csv('clustering_results.csv', index=False)

# Визуализация кластеров (пример для последнего окна)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data[-window_size:])

plt.figure(figsize=(10, 6))
for method, color, marker in zip(['kmeans', 'fcm', 'kohonen'], ['blue', 'green', 'red'], ['o', 's', '^']):
    labels = locals()[f"{method}_labels"]
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis',
               marker=marker, alpha=0.6, label=method)
plt.title('2D проекция кластеров (PCA)')
plt.legend()
plt.show()