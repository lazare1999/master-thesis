import joblib
import pandas as pd
import plotly.graph_objects as go
import visualkeras
from ann_visualizer.visualize import ann_viz
from keras.utils import plot_model
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots


history_tanks = joblib.load("output/history/history_tanks")

def vk(model, name):
    visualkeras.layered_view(model, legend=True, to_file=f"output/plots/{name}_vk.png")

def plot(model, name):
    # Plot model
    plot_model(model, to_file=f'output/plots/{name}.png')

    # Display the image
    data = plt.imread(f'output/plots/{name}.png')
    plt.imshow(data)

def visualize(model, name):
    ann_viz(model, view=True, filename=f"output/plots/{name}.png", title=f"CNN — {name} — Simple Architecture")


def plot_1(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(100)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.show()

def lstm(history):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter( y=history.history['val_loss'], name="val_loss"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter( y=history.history['loss'], name="loss"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter( y=history.history['val_accuracy'], name="val accuracy"),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter( y=history.history['accuracy'], name="accuracy"),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text="Loss/Accuracy of LSTM Model"
    )

    # Set x-axis title
    fig.update_xaxes(title_text="Epoch")

    # Set y-axes titles
    fig.update_yaxes(title_text="<b>primary</b> Loss", secondary_y=False)
    fig.update_yaxes(title_text="<b>secondary</b> Accuracy", secondary_y=True)

    fig.show()