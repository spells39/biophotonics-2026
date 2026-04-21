from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PhotonSample:
    x: np.ndarray
    y: np.ndarray
    r: np.ndarray
    theta: np.ndarray


@dataclass(frozen=True)
class FitResult:
    n_photons: int
    bins: int
    error: float
    max_error: float
    metric: str
    target: str
    seeds: tuple[int, ...]
    require_max_error: bool


def sample_photon_positions(
    n_photons: int,
    a: float,
    rng: np.random.Generator | None = None,
) -> PhotonSample:
    # Базовые проверки
    if n_photons <= 0:
        raise ValueError("Количество фотонов N должно быть положительным.")
    if a <= 0:
        raise ValueError("Параметр пучка a должен быть положительным.")

    # Генератор случайных чисел, если он не был задан
    rng = rng or np.random.default_rng()

    # Заданный профиль интенсивности:
    # p(r) = exp(-(r / sqrt(2a))^2) = exp(-r^2 / (2a)).
    #
    # После нормировки в плоскости XY:
    # rho(x, y) = 1 / (2 pi a) * exp(-(x^2 + y^2) / (2a)).
    #
    # Тогда радиальная плотность имеет вид Рэлея:
    # f(r) = (r / a) * exp(-r^2 / (2a)), r >= 0.

    # Создание набора случайных чисел, из которых получается распределение
    u = rng.random(n_photons)
    # Радиус
    r = np.sqrt(-2.0 * a * np.log(1.0 - u))
    # Подбор угла от 0 до 2 * Pi
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_photons)

    # Перевод в полярные координаты
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Возврат случайных точек
    return PhotonSample(x=x, y=y, r=r, theta=theta)


# Вычисление аналитической двумерной плотности в точках (x,y)
def analytical_density_2d(x: np.ndarray, y: np.ndarray, a: float) -> np.ndarray:
    return np.exp(-(x**2 + y**2) / (2.0 * a)) / (2.0 * np.pi * a)


# Вычисление аналитической плотности на оси y=0
def analytical_axis_density(x: np.ndarray, a: float) -> np.ndarray:
    return analytical_density_2d(x, np.zeros_like(x), a)


# Считает аналитическую плотность только для координаты x
def analytical_x_marginal(x: np.ndarray, a: float) -> np.ndarray:
    return np.exp(-(x**2) / (2.0 * a)) / np.sqrt(2.0 * np.pi * a)


# Возвращает аналитическую плотность радиуса (распределение Рэлея)
def analytical_r_density(r: np.ndarray, a: float) -> np.ndarray:
    return (r / a) * np.exp(-(r**2) / (2.0 * a))


# Считает аналитическую кривую для полосы |y| <= h
def analytical_axis_strip_density(x: np.ndarray, a: float, strip_halfwidth: float) -> np.ndarray:
    if strip_halfwidth <= 0:
        raise ValueError("Полуширина полосы должна быть положительной.")

    # Аргумент для фуцнкции ошибок
    erf_argument = strip_halfwidth / np.sqrt(2.0 * a)
    # Множитель перед exp(), возникающий после интегрирования по y
    strip_factor = math.erf(float(erf_argument)) / (2.0 * strip_halfwidth * np.sqrt(2.0 * np.pi * a))
    # Итоговая аналитическая кривая
    return np.exp(-(x**2) / (2.0 * a)) * strip_factor


# Строит численное осевое сечение
def build_axis_section(
    sample: PhotonSample,
    a: float,
    bins: int = 80,
    extent_sigma: float = 4.0,
    strip_halfwidth_sigma: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Линейный масштаб модели
    sigma = np.sqrt(a)
    limit = extent_sigma * sigma
    strip_halfwidth = strip_halfwidth_sigma * sigma
    # Строит границы бинов по x
    edges = np.linspace(-limit, limit, bins + 1)

    # Строит булеву маску (true - для точек, попавших в полосу около оси; false - для остальных)
    strip_mask = np.abs(sample.y) <= strip_halfwidth
    # Отсеивание фотонов, не лежащих в полосе
    strip_x = sample.x[strip_mask]
    # Разбиение оси x на интервалы, подсчёт количества точек, попавших в каждый интервал (density=False - только количество точек, не плотность)
    counts, x_edges = np.histogram(strip_x, bins=edges, density=False)
    # Центры бинов
    centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    # Ширина одного бина
    dx = x_edges[1] - x_edges[0]

    # Численная оценка двумерной плотности, усреднённой по полосе
    numeric_axis_density = counts / (len(sample.x) * dx * 2.0 * strip_halfwidth)
    # Строит аналитическую кривую для той же полосы
    analytic_axis = analytical_axis_strip_density(centers, a, strip_halfwidth)
    return centers, numeric_axis_density, analytic_axis


# Строит численное одномерное распределение x, сравнивает с аналитической гауссовой кривой
def build_x_marginal(
    sample: PhotonSample,
    a: float,
    bins: int = 100,
    extent_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Масштаб
    sigma = np.sqrt(a)
    limit = extent_sigma * sigma
    # Создание бинов
    edges = np.linspace(-limit, limit, bins + 1)
    # Нормировка гистограммы как плотности вероятности
    hist, edges = np.histogram(sample.x, bins=edges, density=True)
    # Центры бинов
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Построение аналитической кривой
    analytic = analytical_x_marginal(centers, a)
    return centers, hist, analytic


# Строит численное одномерное распределение радиуса (проверяет, что радиус сгенерирован правильно)
def build_r_profile(
    sample: PhotonSample,
    a: float,
    bins: int = 100,
    extent_sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Масштаб
    sigma = np.sqrt(a)
    limit = extent_sigma * sigma
    # Создание бинов
    edges = np.linspace(0.0, limit, bins + 1)
    # Построение нормированной гистограммы радиусов
    hist, edges = np.histogram(sample.r, bins=edges, density=True)
    # Центры бинов
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Построение аналитическрй кривой Рэлея
    analytic = analytical_r_density(centers, a)
    return centers, hist, analytic


# Вычисление средней квадратичной ошибки между двумя массивами
def mean_squared_error(numeric: np.ndarray, analytic: np.ndarray) -> float:
    return float(np.mean((numeric - analytic) ** 2))


# Более "умная" ошибка
def relative_l2_error(numeric: np.ndarray, analytic: np.ndarray) -> float:
    denominator = float(np.sqrt(np.mean(analytic**2)))
    if denominator == 0.0:
        return 0.0
    return float(np.sqrt(np.mean((numeric - analytic) ** 2)) / denominator)


# Выбирает, что нужно сравнивать с аналитикой, считает ошибку (проверяет насколько выборка похожа на теорию)
def compute_profile_error(
    sample: PhotonSample,
    a: float,
    bins: int,
    target: str = "x_marginal",
    metric: str = "relative_l2",
) -> float:
    if target == "x_marginal":
        # Сравниваем численную гистограмму x и аналитическую гауссову кривую
        _, numeric, analytic = build_x_marginal(sample, a, bins=max(20, bins))
    elif target == "r_profile":
        # Сравниваем численную гистограмму радиусов и аналитическое распределение Рэлея
        _, numeric, analytic = build_r_profile(sample, a, bins=max(20, bins))
    elif target == "axis_section":
        # Сравниваем численное осевое сечение по полосе и аналитическую кривую для той же полосы
        _, numeric, analytic = build_axis_section(sample, a, bins=max(20, bins))
    else:
        raise ValueError(
            "Неизвестная цель target. Используйте 'x_marginal', 'r_profile' или 'axis_section'."
        )

    # Выбор метрики
    if metric == "mse":
        return mean_squared_error(numeric, analytic)
    if metric == "relative_l2":
        return relative_l2_error(numeric, analytic)

    raise ValueError("Неизвестная метрика metric. Используйте 'mse' или 'relative_l2'.")


# Автоподбор численных параметров: сколько фотонов брать и сколько бинов использовать
def auto_tune_simulation(
    a: float,
    error_tolerance: float,
    n_candidates: list[int] | tuple[int, ...] = (1000, 2000, 5000, 10000, 20000, 50000),
    bins_candidates: list[int] | tuple[int, ...] = (40, 60, 80, 100, 120, 140),
    target: str = "x_marginal",
    metric: str = "relative_l2",
    seed: int | None = 42,
    n_trials: int = 5,
    require_max_error: bool = True,
) -> FitResult:
    # Базовые проверки
    if a <= 0:
        raise ValueError("Параметр пучка a должен быть положительным.")
    if error_tolerance <= 0:
        raise ValueError("Порог ошибки должен быть положительным.")
    if n_trials <= 0:
        raise ValueError("Количество прогонов n_trials должно быть положительным.")

    # Генерация случайных наборов
    base_seed = 0 if seed is None else seed
    seeds = tuple(base_seed + i for i in range(n_trials))

    # Лучший найденный вариант
    best_result: FitResult | None = None

    # Перебор всех комбинаций N и bins
    for n_photons in n_candidates:
        for bins in bins_candidates:
            errors: list[float] = []
            for current_seed in seeds:
                rng = np.random.default_rng(current_seed)
                sample = sample_photon_positions(n_photons, a, rng=rng)
                error = compute_profile_error(
                    sample=sample,
                    a=a,
                    bins=bins,
                    target=target,
                    metric=metric,
                )
                errors.append(error)

            # Проверка качества наборов
            mean_error = float(np.mean(errors))
            max_error = float(np.max(errors))
            # Информация о текущем лучшем варианте
            candidate = FitResult(
                n_photons=n_photons,
                bins=bins,
                error=mean_error,
                max_error=max_error,
                metric=metric,
                target=target,
                seeds=seeds,
                require_max_error=require_max_error,
            )

            if best_result is None or mean_error < best_result.error:
                best_result = candidate

            if require_max_error:
                is_good_enough = mean_error <= error_tolerance and max_error <= error_tolerance
            else:
                is_good_enough = mean_error <= error_tolerance

            if is_good_enough:
                return candidate

    if best_result is None:
        raise RuntimeError("Не удалось выполнить автоподбор параметров.")

    return best_result


def format_fit_result(result: FitResult, tolerance: float) -> str:
    if result.require_max_error:
        reached = "да" if result.error <= tolerance and result.max_error <= tolerance else "нет"
        criterion = "mean_error <= tolerance и max_error <= tolerance"
    else:
        reached = "да" if result.error <= tolerance else "нет"
        criterion = "mean_error <= tolerance"
    seeds_text = ", ".join(str(seed) for seed in result.seeds)
    return (
        "Автоподбор параметров моделирования\n"
        f"target = {result.target}\n"
        f"metric = {result.metric}\n"
        f"seeds = [{seeds_text}]\n"
        f"criterion = {criterion}\n"
        f"N = {result.n_photons}\n"
        f"bins = {result.bins}\n"
        f"mean_error = {result.error:.6e}\n"
        f"max_error = {result.max_error:.6e}\n"
        f"tolerance = {tolerance:.6e}\n"
        f"Порог достигнут: {reached}"
    )


def save_positions(sample: PhotonSample, output_path: str | Path) -> None:
    output_path = Path(output_path)
    data = np.column_stack((sample.x, sample.y, sample.r, sample.theta))
    header = "x,y,r,theta"
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")


def plot_results(
    sample: PhotonSample,
    a: float,
    bins: int = 140,
    output_path: Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    sigma = np.sqrt(a)
    xy_limit = 4.0 * sigma

    axis_x, axis_numeric, axis_analytic = build_axis_section(
        sample,
        a,
        bins=max(50, bins // 2),
    )
    x_centers, x_numeric, x_analytic = build_x_marginal(sample, a, bins=max(60, bins // 2))
    r_centers, r_numeric, r_analytic = build_r_profile(sample, a, bins=max(60, bins // 2))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    scatter_size = max(3, min(12, int(8000 / len(sample.x)) if len(sample.x) else 3))
    axes[0, 0].scatter(sample.x, sample.y, s=scatter_size, alpha=0.35, linewidths=0)
    axes[0, 0].set_title("Начальные координаты фотонов в плоскости XY")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].set_xlim(-xy_limit, xy_limit)
    axes[0, 0].set_ylim(-xy_limit, xy_limit)
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].plot(r_centers, r_numeric, label="Численный радиальный профиль", lw=2)
    axes[0, 1].plot(r_centers, r_analytic, "--", label="Аналитический профиль", lw=2)
    axes[0, 1].set_title("Распределение радиуса r")
    axes[0, 1].set_xlabel("r")
    axes[0, 1].set_ylabel("Плотность")
    axes[0, 1].grid(alpha=0.2)
    axes[0, 1].legend()

    axes[1, 0].plot(x_centers, x_numeric, label="Численное распределение x", lw=2)
    axes[1, 0].plot(x_centers, x_analytic, "--", label="Аналитическая маргиналь x", lw=2)
    axes[1, 0].set_title("Одномерное распределение по оси x")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("Плотность")
    axes[1, 0].grid(alpha=0.2)
    axes[1, 0].legend()

    axes[1, 1].plot(axis_x, axis_numeric, label="Численное сечение, усредненное по полосе", lw=2)
    axes[1, 1].plot(axis_x, axis_analytic, "--", label="Аналитическая кривая для той же полосы", lw=2)
    axes[1, 1].set_title("Осевое сечение двумерной плотности")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("Плотность")
    axes[1, 1].grid(alpha=0.2)
    axes[1, 1].legend()

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig, axes


def plot_a_comparison(
    a_values: list[float] | tuple[float, ...],
    n_photons: int = 3000,
    bins: int = 100,
    seed: int = 42,
    output_path: Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    if len(a_values) == 0:
        raise ValueError("Нужно передать хотя бы одно значение a.")

    fig, axes = plt.subplots(len(a_values), 2, figsize=(12, 4.2 * len(a_values)))
    if len(a_values) == 1:
        axes = np.array([axes])

    for row_index, a in enumerate(a_values):
        rng = np.random.default_rng(seed)
        sample = sample_photon_positions(n_photons, a, rng=rng)
        sigma = np.sqrt(a)
        limit = 4.0 * sigma

        scatter_size = max(4, min(12, int(7000 / n_photons)))
        axes[row_index, 0].scatter(sample.x, sample.y, s=scatter_size, alpha=0.35, linewidths=0)
        axes[row_index, 0].set_title(f"Координаты фотонов, a = {a}")
        axes[row_index, 0].set_xlabel("x")
        axes[row_index, 0].set_ylabel("y")
        axes[row_index, 0].set_xlim(-limit, limit)
        axes[row_index, 0].set_ylim(-limit, limit)
        axes[row_index, 0].set_aspect("equal", adjustable="box")
        axes[row_index, 0].grid(alpha=0.2)

        axis_x, axis_numeric, axis_analytic = build_axis_section(sample, a, bins=max(50, bins // 2))
        axes[row_index, 1].plot(axis_x, axis_numeric, lw=2, label="Численное сечение")
        axes[row_index, 1].plot(axis_x, axis_analytic, "--", lw=2, label="Аналитическая кривая")
        axes[row_index, 1].set_title(f"Осевое сечение, a = {a}")
        axes[row_index, 1].set_xlabel("x")
        axes[row_index, 1].set_ylabel("Плотность")
        axes[row_index, 1].grid(alpha=0.2)
        axes[row_index, 1].legend()

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig, axes


def summarize_sample(sample: PhotonSample, a: float) -> str:
    mean_r = float(np.mean(sample.r))
    std_x = float(np.std(sample.x))
    mean_x = float(np.mean(sample.x))
    theoretical_mean_r = float(np.sqrt(np.pi * a / 2.0))
    theoretical_std_x = float(np.sqrt(a))

    return (
        f"N = {len(sample.x)}\n"
        f"a = {a}\n"
        f"Средний радиус: {mean_r:.6f} (теория: {theoretical_mean_r:.6f})\n"
        f"Среднее по x: {mean_x:.6f} (теория: 0.000000)\n"
        f"СКО по x: {std_x:.6f} (теория: {theoretical_std_x:.6f})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Генерация случайных координат фотонов в осесимметричном пучке "
            "с профилем p(r) = exp(-(r / sqrt(2a))^2)."
        )
    )
    parser.add_argument("--N", type=int, required=True, help="Количество фотонов.")
    parser.add_argument("--a", type=float, required=True, help="Параметр пучка.")
    parser.add_argument(
        "--bins",
        type=int,
        default=140,
        help="Количество бинов для построения плотностей.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Зерно генератора случайных чисел.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("photon_positions.csv"),
        help="Путь для сохранения координат фотонов в CSV.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("photon_beam.png"),
        help="Путь для сохранения итоговой фигуры.",
    )
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Включить автоподбор N и bins по заданному порогу ошибки.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5e-2,
        help="Допустимый порог ошибки для автоподбора.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="x_marginal",
        choices=("x_marginal", "r_profile", "axis_section"),
        help="Какая аналитическая зависимость используется в автоподборе.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="relative_l2",
        choices=("relative_l2", "mse"),
        help="Метрика ошибки для автоподбора.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Число прогонов с разными seed для устойчивого автоподбора.",
    )
    parser.add_argument(
        "--mean-only",
        action="store_true",
        help="Использовать более мягкий критерий: учитывать только среднюю ошибку.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.auto_tune:
        fit_result = auto_tune_simulation(
            a=args.a,
            error_tolerance=args.tolerance,
            target=args.target,
            metric=args.metric,
            seed=args.seed,
            n_trials=args.trials,
            require_max_error=not args.mean_only,
        )
        print(format_fit_result(fit_result, args.tolerance))
        print()
        args.N = fit_result.n_photons
        args.bins = fit_result.bins

    rng = np.random.default_rng(args.seed)
    sample = sample_photon_positions(args.N, args.a, rng=rng)
    save_positions(sample, args.csv)
    plot_results(sample, a=args.a, bins=args.bins, output_path=args.figure)

    print(f"Сохранены координаты фотонов: {args.csv.resolve()}")
    print(f"Сохранен график: {args.figure.resolve()}")
    print()
    print(summarize_sample(sample, args.a))


if __name__ == "__main__":
    main()
