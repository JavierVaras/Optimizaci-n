import random as rnd
import math
import matplotlib.pyplot as plt

class Problem:
    def __init__(self):
        self.dimension = 5  # Cambiar, dependiendo del problema

    def eval(self, x):
        # Función objetivo escalarizada
        return ((70 * x[0] + 92 * x[1] + 50 * x[2] + 65 * x[3] + 25 * x[4]) / 3360) * 0.7 + ((1000 - 170 * x[0] + 320 * x[1] + 60 * x[2] + 105 * x[3] + 15 * x[4]) / 1000) * 0.3

    def check(self, x):
        # Se verifica que los valores de las variables cumplan con las restricciones
        if x[0] > 15 or x[1] > 10 or x[2] > 25 or x[3] > 4 or x[4] > 30:
            return False
        if 170 * x[0] + 320 * x[1] > 3800:
            return False
        if 60 * x[2] > 2800:
            return False
        if 105 * x[3] > 2800:
            return False
        if 60 * x[2] + 15 * x[4] > 3500:
            return False
        return True

class Agent(Problem):
    def __init__(self):
        super().__init__()
        self.p = Problem()
        self.x = [rnd.randint(0, 30) for _ in range(self.p.dimension)]
        while not self.isFeasible():  # Se verifica el cumplimiento de las restricciones
            self.x = [rnd.randint(0, 30) for _ in range(self.p.dimension)]

    def isFeasible(self):
        return self.p.check(self.x)

    def isBetterThan(self, g):
        return self.fit() > g.fit()  # Cambiar según el criterio de optimización (maximizar)

    def fit(self):
        return self.p.eval(self.x)

    def move(self, g, p=0.8):
        new_x = self.x.copy()
        
        if rnd.random() < p:
            # Búsqueda global
            r = rnd.random()
            for j in range(self.p.dimension):
                new_x[j] = self.x[j] + (g.x[j] - self.x[j]) * r * r
        else:
            # Búsqueda local
            r = rnd.random()
            l = rnd.random()
            for j in range(self.p.dimension):
                new_x[j] = self.x[j] + (r * r * self.x[j] - g.x[j]) * l * (1 - l) + (g.x[j] - self.x[j]) * l

        # Ajustar valores para que cumplan con las restricciones
        new_x = self.adjust(new_x)

        # Aplicar la función sigmoide para obtener valores binarios
        self.x = [self.toBinary(val) for val in new_x]

    def toBinary(self, x):
        return 1 if (1 / (1 + math.exp(-x))) > rnd.random() else 0

    def adjust(self, x):
        # Asegurar que las variables estén dentro de un rango válido
        x = [max(0, min(30, val)) for val in x]

        # Aplicar la corrección de restricciones
        while not self.p.check(x):
            for j in range(self.p.dimension):
                # Corrección simple por ajuste en el rango
                if x[0] > 15:
                    x[0] = 15
                if x[1] > 10:
                    x[1] = 10
                if x[2] > 25:
                    x[2] = 25
                if x[3] > 4:
                    x[3] = 4
                if x[4] > 30:
                    x[4] = 30

                # Ajuste para cumplir con restricciones adicionales
                if 170 * x[0] + 320 * x[1] > 3800:
                    x[1] = max(0, 3800 - 170 * x[0]) / 320
                if 60 * x[2] > 2800:
                    x[2] = 2800 / 60
                if 105 * x[3] > 2800:
                    x[3] = 2800 / 105
                if 60 * x[2] + 15 * x[4] > 3500:
                    x[4] = max(0, (3500 - 60 * x[2]) / 15)
            
        return x

    def __str__(self) -> str:
        return f"fit:{self.fit()} x:{self.x}"

    def copy(self, a):
        self.x = a.x.copy()

class Swarm:
    def __init__(self):
        self.maxIter = 50
        self.nAgents = 5  # Número de agentes
        self.swarm = []
        self.g = Agent()
        self.best_fit_per_iter = []  # Almacenar los valores de la función objetivo

    def solve(self):
        self.initRand()
        self.evolve()
        self.plot_results()

    def initRand(self):
        for _ in range(self.nAgents):
            while True:
                a = Agent()
                if a.isFeasible():
                    break
            self.swarm.append(a)

        self.g.copy(self.swarm[0])
        for i in range(1, self.nAgents):
            if self.swarm[i].isBetterThan(self.g):
                self.g.copy(self.swarm[i])

        self.toConsole(0)  # Iniciar consola con la primera iteración
        self.best_fit_per_iter.append(self.g.fit())  # Almacenar el valor de la función objetivo

    def evolve(self):
        t = 1
        while t <= self.maxIter:
            for i in range(self.nAgents):
                a = Agent()
                while True:
                    a.copy(self.swarm[i])
                    a.move(self.g)
                    if a.isFeasible():
                        break
                self.swarm[i].copy(a)

            for i in range(self.nAgents):
                if self.swarm[i].isBetterThan(self.g):
                    self.g.copy(self.swarm[i])

            self.toConsole(t)
            self.best_fit_per_iter.append(self.g.fit())  # Almacenar el valor de la función objetivo
            t += 1

    def toConsole(self, t):
        print(f"Iteración {t}: Mejor agente: {self.g}")

    def plot_results(self):
        plt.figure(figsize=(12, 6))

        # Gráfico de la evolución del mejor valor de la función objetivo
        plt.subplot(1, 2, 1)
        plt.plot(self.best_fit_per_iter, marker='o', linestyle='-', color='b')
        plt.xlabel('Iteración')
        plt.ylabel('Mejor valor de la función objetivo')
        plt.title('Evolución del valor de la función objetivo')
        plt.grid(True)

        # Diagrama de caja de los valores de la función objetivo
        plt.subplot(1, 2, 2)
        plt.boxplot(self.best_fit_per_iter, vert=True, patch_artist=True)
        plt.xlabel('Valores de la función objetivo')
        plt.title('Diagrama de caja de los valores de la función objetivo')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

try:
    Swarm().solve()
except Exception as e:
    print(f"{e} \nCaused by {e.__cause__}")