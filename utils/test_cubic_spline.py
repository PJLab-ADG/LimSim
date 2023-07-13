import matplotlib.pyplot as plt
import numpy as np

from utils.cubic_spline import Spline, Spline2D

def main():
    # test spline
    x = np.array([0, 1, 2, 3])
    y = np.exp([0, 1, 2, 3])
    spline = Spline(x_list=x, y_list=y)

    print(np.allclose(spline.a, y))
    print(np.allclose(spline.c, np.array([0, 0.75685, 5.83007, 0]), atol=1e-5))
    print(
        np.allclose(spline.b, np.array([1.46600, 2.22285, 8.80977]), atol=1e-5))
    print(
        np.allclose(spline.d, np.array([0.25228, 1.69107, -1.94336]),
                    atol=1e-5))

    ds = 0.01
    x_list = np.arange(spline.x_list[0], spline.x_list[-1], ds)
    y_list = np.array([spline.calculate_approximation(x) for x in x_list])
    plt.plot(x, y, '*', label="input")
    plt.plot(x_list, y_list, label="spline")
    plt.legend()


    # test Spline2D
    x = [-2.5, 0.0, 2.5, 5.0, 7.5, 3.0, -1.0]
    y = [0.7, -6, 5, 6.5, 0.0, 5.0, -2.0]
    ds = 0.01  # [m] distance of each interpolated points

    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    plt.subplots(1)
    plt.plot(x, y, "*", label="input")
    plt.plot(rx, ry, "-r", label="spline")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.legend()

    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
    ax1.grid(True)
    ax1.legend()
    ax1.set_xlabel("line length[m]")
    ax1.set_ylabel("yaw angle[deg]")

    ax2.plot(s, rk, "-r", label="curvature")
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlabel("line length[m]")
    ax2.set_ylabel("curvature [1/m]")

    plt.show()


if __name__ == "__main__":
    main()