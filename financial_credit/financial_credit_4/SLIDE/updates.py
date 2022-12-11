import csv


""" DI """

def update_perfs_di(perfs, learning_stats) :

    acc, bacc, di = perfs

    learning_stats["acc"].append(acc)
    learning_stats["bacc"].append(bacc)
    learning_stats["di"].append(di)

    return learning_stats


def write_perfs_di(result_path, file_name, mode, lmda, tau, learning_stats) :

    file = open(result_path + file_name, "a")
    writer = csv.writer(file)
    writer.writerow([mode, lmda, tau,
                    learning_stats["acc"][-1], learning_stats["bacc"][-1], learning_stats["di"][-1]]
                   )
    file.close()
