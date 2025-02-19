import numpy as np
import matplotlib.pyplot as plt



def f(x):
	return np.sin(x) + 2 * x**2

def exact_derivative(x):
	return np.cos(x) + 4*x;

def forward(f, x, h):
	return (f(x+h) - f(x))/h;


def backward(f, x, h):
	return (f(x)-f(x-h))/h

def central(f, x, h):
	return (f(x+h) - f(x-h))/(2*h);

def plot_derivatives(x_values, exact, forward, backward, central):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, exact, label='Exact Derivative', color='black', linewidth=2)
    plt.plot(x_values, forward, label='Forward Difference', linestyle='--')
    plt.plot(x_values, backward, label='Backward Difference', linestyle='--')
    plt.plot(x_values, central, label='Central Difference', linestyle='--')
    plt.title('Comparison of Finite Difference Methods for Derivative Approximation')
    plt.xlabel('x')
    plt.ylabel('Derivative')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
	h = 0.05
	user_input = input("Currently h = {h}. Would you like to change it? (y/n): ")
	if user_input.lower() == "y":
		try:
			h = float(input("Enter new h: "))
		except ValueError:
			print("INvalid input. Using default h = 0.05")


	x_values = np.linspace(0, 1, 100)

	exact_values = exact_derivative(x_values)

	forward_values = forward(f, x_values, h)
	backward_values = backward(f, x_values, h)
	central_values = central(f, x_values, h)

	plot_derivatives(x_values, exact_values, forward_values, backward_values, central_values)


if __name__ == "__main__":
	main()









