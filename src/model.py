import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import shap

def entrenar_modelo(X_train, y_train, n_estimators=100, random_state=42):
    """
    Entrena un clasificador Random Forest.

    Args:
        X_train (pd.DataFrame): Características de entrenamiento.
        y_train (np.ndarray): Etiquetas codificadas de entrenamiento.
        n_estimators (int, optional): Número de árboles. Por defecto 100.
        random_state (int, optional): Semilla para reproducibilidad.
            Por defecto 42.

    Returns:
        RandomForestClassifier: Modelo entrenado.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluar_modelo(model, X_test, y_test, target_names=None):
    """
    Evalúa el modelo en test e imprime el classification report.

    Calcula el accuracy y muestra el classification report ajustando los
    nombres de clase a las clases realmente presentes en y_test.

    Args:
        model: Modelo entrenado.
        X_test (pd.DataFrame): Características de test.
        y_test (np.ndarray): Etiquetas reales de test.
        target_names (list[str], optional): Nombres originales de las clases
            (ordenados por índice de codificación). Si es None, se usan
            índices numéricos.

    Returns:
        tuple: (y_pred, acc)
            - y_pred (np.ndarray): Predicciones del modelo.
            - acc (float): Accuracy en test.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy en test: {acc:.4f}")

    # Obtener clases únicas presentes en y_test
    unique_labels = np.unique(y_test)

    # Si target_names es None, usamos None (classification_report usará números)
    if target_names is None:
        print("\nClassification Report (sin nombres de clase):")
        print(classification_report(y_test, y_pred))
    else:
        # Filtrar target_names para que coincida con las clases presentes
        target_names_subset = [target_names[i] for i in unique_labels]
        print("\nClassification Report (clases en test):")
        print(classification_report(y_test, y_pred,
                                    labels=unique_labels,
                                    target_names=target_names_subset))
    return y_pred, acc

def validacion_cruzada(model, X_train, y_train, cv=5, scoring='accuracy'):
    """
    Realiza validación cruzada estratificada y devuelve media y desviación.

    Args:
        model: Modelo a evaluar.
        X_train (pd.DataFrame): Características de entrenamiento.
        y_train (np.ndarray): Etiquetas de entrenamiento.
        cv (int, optional): Número de folds. Por defecto 5.
        scoring (str, optional): Métrica de evaluación. Por defecto 'accuracy'.

    Returns:
        tuple: (media, desviacion)
            - media (float): Media de la métrica en los folds.
            - desviacion (float): Desviación estándar de la métrica.
    """
    cv_strat = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv_strat, scoring=scoring)
    media = scores.mean()
    desv = scores.std()
    print(f"CV {scoring} media: {media:.4f} (+/- {desv:.4f})")
    return media, desv

def guardar_modelo(model, ruta):
    """
    Guarda un modelo en disco con joblib.

    Args:
        model: Modelo a guardar.
        ruta (str): Ruta del archivo de salida (.joblib).

    Returns:
        None
    """
    joblib.dump(model, ruta)
    print(f"Modelo guardado en {ruta}")

def cargar_modelo(ruta):
    """
    Carga un modelo guardado con joblib.

    Args:
        ruta (str): Ruta del archivo .joblib.

    Returns:
        El modelo cargado.
    """
    return joblib.load(ruta)

def plot_waterfall(shap_values_3d, X_sample, idx, clase_idx, feature_names, expected_value, top_n=12, title=""):
    """
    Genera un waterfall plot con los SHAP values de una clase específica.

    Selecciona las top_n características con mayor valor absoluto de SHAP
    para la muestra y clase indicadas, y las representa como barras
    horizontales (rojas si empujan hacia la clase, azules si la alejan).

    Args:
        shap_values_3d (np.ndarray): Array de forma (n_muestras, n_features, n_clases).
        X_sample (pd.DataFrame): DataFrame con las muestras.
        idx (int): Índice de la muestra a explicar.
        clase_idx (int): Índice de la clase a explicar.
        feature_names (list[str]): Nombres de las características.
        expected_value (np.ndarray): Valor base esperado por clase.
        top_n (int, optional): Número de características a mostrar. Por defecto 12.
        title (str, optional): Título del gráfico. Por defecto "".

    Returns:
        tuple: (fig, ax) Figura y ejes de matplotlib.
    """
    shap_muestra = shap_values_3d[idx, :, clase_idx]
    top_n = min(top_n, len(shap_muestra))
    indices = np.argsort(np.abs(shap_muestra))[::-1][:top_n]
    shap_top = shap_muestra[indices]
    names_top = [feature_names[i] for i in indices]
    colores = ['red' if v > 0 else 'blue' for v in shap_top]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n*0.5)))
    y_pos = np.arange(len(shap_top))
    ax.barh(y_pos, shap_top, color=colores, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_top, fontsize=10)
    ax.set_xlabel('SHAP value')

    max_abs = np.max(np.abs(shap_top)) if len(shap_top) > 0 else 1
    for i, v in enumerate(shap_top):
        offset = 0.02 * max_abs
        ax.text(v + offset if v >= 0 else v - offset, i, f'{v:.3f}',
                va='center', ha='left' if v >= 0 else 'right', fontsize=9)

    ax.set_title(f"{title} - Valor base: {expected_value[clase_idx]:.3f}", fontsize=12)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', label='Empuja a la clase'),
                       Patch(facecolor='blue', label='Aleja de la clase')]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    return fig, ax

def calcular_shap_values(model, X_sample):
    """
    Calcula los SHAP values de un modelo Random Forest.

    Args:
        model: Modelo de clasificación entrenado (Random Forest).
        X_sample (pd.DataFrame): Muestras sobre las que calcular SHAP.

    Returns:
        tuple: (explainer, shap_values)
            - explainer: Objeto TreeExplainer de SHAP.
            - shap_values (np.ndarray): Array de forma
              (n_muestras, n_features, n_clases) con los SHAP values.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)  # (n_muestras, n_features, n_clases)
    return explainer, shap_values
