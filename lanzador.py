
from FuncionesAux import *
import warnings
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
"""
import gc
gc.collect()
"""

warnings.filterwarnings("ignore")
import multiprocessing
import time

print("Number of cpu : ", multiprocessing.cpu_count())

# limpiar_csv(5)
rondas = 20
n_clientes_grid = [300]  # ,200,250,300]#, 250, 200, 150, 100, 60,50,40,30,20,10]
# n_clientes = 30
acc_media_gb = []
# for i in n_clientes:

# n_clientes = n_clientes
clients = []
"""
dir_base = os.getcwd()
df = pd.DataFrame(columns=["clientes", "accuracy", "tiempo"])
df.to_csv(dir_base+"/grid_acc_medias_fp_tiempo.csv",index=False)
"""
for n_clientes in n_clientes_grid:
    inicio = time.time()
    server = multiprocessing.Process(target=start_server, args=(n_clientes, rondas))
    server.start()
    time.sleep(10)

    for i in range(n_clientes):
        inx = i + 1
        p = multiprocessing.Process(target=start_client, args=[inx])
        p.start()
        clients.append(p)

    server.join()
    for client in clients:
        client.join()
    fin = time.time()
    tiempo = fin - inicio
    del fin, inicio

    print("----FL TERMINADO-----")
    time.sleep(20)

    dir_base = os.getcwd()
    dir_carp_graficas = "/metricas_parties"
    dir_graficas = dir_base + dir_carp_graficas

    acc_media = []

    rango = range(1, n_clientes + 1)
    for i in rango:
        party = pd.read_csv(dir_graficas + "/history_" + str(i) + ".csv")
        acc = np.array(party["Accuracy"])
        acc_media.append(acc[-1])

    # acc_media_gb.append(sum(acc_media)/len(acc_media))
    acc_media_gb = sum(acc_media) / len(acc_media)

    fd_aux = [[n_clientes], [acc_media_gb], [tiempo]]
    fd_aux = np.transpose(fd_aux)
    df = pd.DataFrame(fd_aux, columns=["clientes", "accuracy", "tiempo"])

    base = pd.read_csv(dir_base + "/grid_acc_medias_fp_tiempo.csv")
    base = base.append(df)

    base.to_csv(dir_base + "/grid_acc_medias_fp_tiempo.csv", index=False)
