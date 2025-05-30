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
    np.random.seed(42)
    # Реальные параметры руды КМА:
    cluster1 = np.random.normal(loc=[35, 20, 40], scale=[3, 2, 5], size=(n_samples//3, 3))  # Бедная руда
    cluster2 = np.random.normal(loc=[45, 30, 55], scale=[4, 3, 7], size=(n_samples//3, 3))  # Средняя
    cluster3 = np.random.normal(loc=[25, 40, 70], scale=[5, 4, 10], size=(n_samples//3, 3)) # Трудноизмельчаемая
    return np.vstack((cluster1, cluster2, cluster3))


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
    fcm = FCM(n_clusters=n_clusters, m=1.5, max_iter=200, random_state=42)
    fcm.fit(data)
    labels = fcm.predict(data)
    return labels, fcm.centers


def kohonen_cluster(data, n_clusters):
    # 1. Параметры карты
    map_size = (int(np.sqrt(5 * n_clusters)), int(np.sqrt(5 * n_clusters)))

    # 2. Инициализация и обучение SOM
    som = MiniSom(map_size[0], map_size[1], data.shape[1],
                  sigma=1.5,
                  learning_rate=0.5,
                  neighborhood_function='gaussian',
                  random_seed=42)

    # 3. Инициализация весов (убираем PCA инициализацию, чтобы избежать комплексных чисел)
    som.random_weights_init(data)

    # 4. Обучение
    print("Training SOM...")
    som.train_random(data, 1000, verbose=True)

    # 5. Получаем координаты победителей для всех точек
    winner_coords = np.array([som.winner(x) for x in data])

    # 6. Кластеризация нейронов
    weights = som.get_weights().reshape(-1, data.shape[1])

    # Используем K-means для кластеризации нейронов
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    neuron_clusters = kmeans.fit_predict(weights)

    # 7. Назначение меток данным
    neuron_indices = winner_coords[:, 0] * map_size[1] + winner_coords[:, 1]
    labels = neuron_clusters[neuron_indices.astype(int)]

    # 8. Вычисление центроидов (с обработкой пустых кластеров)
    centers = []
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            centers.append(cluster_points.mean(axis=0))
        else:
            # Если кластер пустой, используем случайную точку данных
            centers.append(data[np.random.randint(0, len(data))])
    centers = np.array(centers)

    # 9. Визуализация карты с кластерами (исправленная)
    plt.figure(figsize=(10, 10))
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.colorbar()

    # Упрощенная визуализация без маркеров-цифр
    for i in range(len(data)):
        x, y = winner_coords[i]
        plt.scatter(x + 0.5, y + 0.5, color=plt.cm.tab10(labels[i] % 10), s=30)

    plt.title('Карта SOM с кластерами')
    plt.savefig('som_clusters.png')
    plt.close()

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

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))  # Нормализуем в диапазон [0,1] для наглядности
scaled_data = scaler.fit_transform(data)


def plot_3d_clusters(data, labels, title, save_name):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Используем другой цветовой градиент
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                         c=labels, cmap='plasma', s=40, alpha=0.8, edgecolor='k')

    ax.set_xlabel('Содержание железа (α, %)', fontsize=12, labelpad=10)
    ax.set_ylabel('Трудноизмельчаемые (μ, %)', fontsize=12, labelpad=10)
    ax.set_zlabel('Крупность (d, мм)', fontsize=12, labelpad=10)

    ax.set_title(title, fontsize=14, pad=20)
    ax.grid(True)

    # Добавляем цветовую шкалу
    cbar = plt.colorbar(scatter, pad=0.1)
    cbar.set_label('Номер кластера', rotation=270, labelpad=15)

    # Устанавливаем угол обзора
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(f'{save_name}.png', dpi=200, bbox_inches='tight')
    plt.show()


# Пример использования для последнего окна данных
window_data = scaled_data[-window_size:]

# Для k-means
kmeans_labels, _ = kmeans_cluster(window_data, n_clusters)
plot_3d_clusters(window_data, kmeans_labels,
                 "3D-визуализация кластеров (k-means)", "kmeans_3d")

# Для fuzzy c-means
fcm_labels, _ = fuzzy_cmeans(window_data, n_clusters)
plot_3d_clusters(window_data, fcm_labels,
                 "3D-визуализация кластеров (fuzzy c-means)", "fcm_3d")

# Для Kohonen
kohonen_labels, _ = kohonen_cluster(window_data, n_clusters)
plot_3d_clusters(window_data, kohonen_labels,
                 "3D-визуализация кластеров (Kohonen)", "kohonen_3d")