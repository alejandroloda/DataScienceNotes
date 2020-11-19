"""Data Science funtions"""


def confusion_matrix(model, X_test, y_test, labels=None):
    """Fast access to a Confusion Matrix with matplot and sklearn"""
    import matplotlib.pyplot as plt
    from sklearn.metrics import plot_confusion_matrix
    return plot_confusion_matrix(
        model,                                # clasificador/modelo
        X_test,                               # Entrada X del modelo,
        y_test,                               # Entrada Y del modelo
        display_labels=labels, # Etiquetas de las columnas
        #normalize='true',                    # Si se desea normalizada
        cmap=plt.cm.Blues)                    # Colores"
