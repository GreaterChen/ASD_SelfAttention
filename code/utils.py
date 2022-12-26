import matplotlib.pyplot as plt


def draw_result_pic(save_path:str, res:list, start_epoch:int, pic_title:str):
    x = [idx for idx in range(len(res[0]))]
    y0 = res[0]     # train
    y1 = res[1]     # test

    plt.figure()
    plt.plot(x[start_epoch:], y0[start_epoch:], label='train', c='blue')
    plt.plot(x[start_epoch:], y1[start_epoch:], label='test', c='red')
    plt.xlabel('epoch')
    plt.title(pic_title)
    plt.legend()
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.show()

