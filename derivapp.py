import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, latex, simplify, expand, factor
from sympy.parsing.sympy_parser import parse_expr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def crear_ejemplos_funciones():
    """
    Retorna una lista de funciones de ejemplo
    """
    return {
        "Polinomiales": [
            "x**2",
            "x**3 + 2*x**2 - 5*x + 1",
            "2*x**4 - 3*x**3 + x**2 - 7"
        ],
        "Exponenciales": [
            "exp(x)",
            "2*exp(x) + 3",
            "exp(2*x) - exp(-x)"
        ],
        "Logarítmicas": [
            "log(x)",
            "log(x**2 + 1)",
            "x*log(x)"
        ],
        "Trigonométricas": [
            "sin(x)",
            "cos(x)",
            "tan(x)",
            "sin(x)*cos(x)",
            "sin(x**2)"
        ],
        "Mixtas": [
            "x*exp(x)",
            "sin(x)/x",
            "exp(x)*cos(x)",
            "log(x)*sin(x)"
        ]
    }

def validar_funcion(funcion_str):
    """
    Valida si la función ingresada es válida
    """
    try:
        x = symbols('x')
        expr = parse_expr(funcion_str)
        return True, expr
    except Exception as e:
        return False, str(e)

def calcular_derivada(expr, orden=1):
    """
    Calcula la derivada de orden n de una expresión
    """
    x = symbols('x')
    try:
        derivada = diff(expr, x, orden)
        return derivada
    except Exception as e:
        return None

def evaluar_funcion(expr, x_vals):
    """
    Evalúa una función en un conjunto de valores
    """
    x = symbols('x')
    try:
        func = sp.lambdify(x, expr, 'numpy')
        return func(x_vals)
    except:
        return None

def crear_grafico_interactivo(expr, derivadas, x_min, x_max):
    """
    Crea un gráfico interactivo usando Plotly
    """
    x = symbols('x')
    x_vals = np.linspace(x_min, x_max, 1000)
    
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=["Función y sus Derivadas"]
    )
    
    colores = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Función original
    try:
        y_vals = evaluar_funcion(expr, x_vals)
        if y_vals is not None:
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines',
                name=f'f(x) = {expr}',
                line=dict(color=colores[0], width=2)
            ))
    except:
        pass
    
    # Derivadas
    for i, (orden, derivada) in enumerate(derivadas.items()):
        if derivada is not None:
            try:
                y_vals = evaluar_funcion(derivada, x_vals)
                if y_vals is not None:
                    nombre_derivada = f"f'(x)" if orden == 1 else f"f^({orden})(x)"
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=y_vals,
                        mode='lines',
                        name=f'{nombre_derivada} = {derivada}',
                        line=dict(color=colores[i+1], width=2, dash='dash' if i > 0 else 'solid')
                    ))
            except:
                pass
    
    fig.update_layout(
        title="Gráfico de la Función y sus Derivadas",
        xaxis_title="x",
        yaxis_title="y",
        hovermode='x unified',
        showlegend=True
    )
    
    return fig

def mostrar_reglas_derivacion():
    """
    Muestra las reglas básicas de derivación
    """
    st.markdown("""
    ### 📚 Reglas Básicas de Derivación
    
    **Reglas Básicas:**
    - $(c)' = 0$ (constante)
    - $(x^n)' = nx^{n-1}$ (regla de la potencia)
    - $(e^x)' = e^x$ (exponencial)
    - $(ln(x))' = \\frac{1}{x}$ (logaritmo natural)
    
    **Funciones Trigonométricas:**
    - $(sin(x))' = cos(x)$
    - $(cos(x))' = -sin(x)$
    - $(tan(x))' = sec^2(x)$
    
    **Reglas de Operación:**
    - $(f + g)' = f' + g'$ (suma)
    - $(f \\cdot g)' = f'g + fg'$ (producto)
    - $(\\frac{f}{g})' = \\frac{f'g - fg'}{g^2}$ (cociente)
    - $(f(g(x)))' = f'(g(x)) \\cdot g'(x)$ (cadena)
    """)

def main():
    st.title("🧮 Calculadora de Derivadas - Análisis Matemático Univariado")
    st.markdown("---")
    
    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        ### Cómo usar:
        1. Ingresa una función matemática
        2. Selecciona el orden de derivación
        3. Ajusta el rango del gráfico
        4. Visualiza resultados y gráficos
        
        ### Sintaxis:
        - Potencias: `x**2`, `x**3`
        - Exponencial: `exp(x)`
        - Logaritmo: `log(x)`
        - Trigonométricas: `sin(x)`, `cos(x)`, `tan(x)`
        - Constantes: `pi`, `e`
        """)
    
    # Selección de ejemplo o función personalizada
    st.subheader("🔢 Selecciona o ingresa una función")
    
    modo = st.radio("Modo de entrada:", ["Función personalizada", "Ejemplos predefinidos"])
    
    if modo == "Ejemplos predefinidos":
        ejemplos = crear_ejemplos_funciones()
        categoria = st.selectbox("Categoría:", list(ejemplos.keys()))
        funcion_str = st.selectbox("Función:", ejemplos[categoria])
    else:
        funcion_str = st.text_input(
            "Función f(x):",
            value="x**2 + 2*x + 1",
            help="Ejemplo: x**2 + 2*x + 1, sin(x), exp(x), log(x)"
        )
    
    # Validar función
    es_valida, resultado = validar_funcion(funcion_str)
    
    if not es_valida:
        st.error(f"❌ Error en la función: {resultado}")
        return
    
    expr = resultado
    
    # Configuración de derivadas
    st.subheader("⚙️ Configuración")
    
    col1, col2 = st.columns(2)
    
    with col1:
        orden_max = st.slider("Orden máximo de derivación:", 1, 5, 3)
        
    with col2:
        evaluar_punto = st.checkbox("Evaluar en un punto específico")
        if evaluar_punto:
            punto = st.number_input("Punto de evaluación:", value=1.0)
    
    # Calcular derivadas
    derivadas = {}
    for orden in range(1, orden_max + 1):
        derivada = calcular_derivada(expr, orden)
        derivadas[orden] = derivada
    
    # Mostrar resultados
    st.subheader("📊 Resultados")
    
    # Función original
    st.markdown(f"**Función original:** $f(x) = {latex(expr)}$")
    
    # Derivadas
    for orden, derivada in derivadas.items():
        if derivada is not None:
            nombre = f"f'(x)" if orden == 1 else f"f^{{({orden})}}(x)"
            st.markdown(f"**Derivada de orden {orden}:** ${nombre} = {latex(derivada)}$")
            
            # Simplificar si es posible
            derivada_simplificada = simplify(derivada)
            if derivada_simplificada != derivada:
                st.markdown(f"**Simplificada:** ${nombre} = {latex(derivada_simplificada)}$")
    
    # Evaluación en punto específico
    if evaluar_punto:
        st.subheader(f"🎯 Evaluación en x = {punto}")
        
        x = symbols('x')
        try:
            valor_func = float(expr.subs(x, punto))
            st.write(f"f({punto}) = {valor_func:.6f}")
            
            for orden, derivada in derivadas.items():
                if derivada is not None:
                    valor_deriv = float(derivada.subs(x, punto))
                    nombre = f"f'({punto})" if orden == 1 else f"f^({orden})({punto})"
                    st.write(f"{nombre} = {valor_deriv:.6f}")
        except Exception as e:
            st.error(f"Error al evaluar en el punto: {e}")
    
    # Configuración del gráfico
    st.subheader("📈 Visualización")
    
    col1, col2 = st.columns(2)
    with col1:
        x_min = st.number_input("Valor mínimo de x:", value=-5.0)
    with col2:
        x_max = st.number_input("Valor máximo de x:", value=5.0)
    
    if x_min >= x_max:
        st.error("El valor mínimo debe ser menor que el máximo")
        return
    
    # Crear y mostrar gráfico
    try:
        fig = crear_grafico_interactivo(expr, derivadas, x_min, x_max)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error al crear el gráfico: {e}")
    
    # Análisis adicional
    st.subheader("🔍 Análisis Adicional")
    
    with st.expander("Análisis de la función"):
        x = symbols('x')
        
        # Puntos críticos (donde f'(x) = 0)
        if 1 in derivadas and derivadas[1] is not None:
            try:
                puntos_criticos = sp.solve(derivadas[1], x)
                if puntos_criticos:
                    st.write("**Puntos críticos (f'(x) = 0):**")
                    for pc in puntos_criticos:
                        if pc.is_real:
                            st.write(f"x = {pc}")
                else:
                    st.write("No se encontraron puntos críticos reales")
            except:
                st.write("No se pudieron calcular los puntos críticos")
        
        # Información sobre concavidad
        if 2 in derivadas and derivadas[2] is not None:
            st.write("**Segunda derivada (concavidad):**")
            st.write(f"f''(x) = {derivadas[2]}")
            
            try:
                puntos_inflexion = sp.solve(derivadas[2], x)
                if puntos_inflexion:
                    st.write("**Puntos de inflexión (f''(x) = 0):**")
                    for pi in puntos_inflexion:
                        if pi.is_real:
                            st.write(f"x = {pi}")
            except:
                st.write("No se pudieron calcular los puntos de inflexión")
    
    # Reglas de derivación
    with st.expander("📖 Reglas de Derivación"):
        mostrar_reglas_derivacion()
    
    # Información técnica
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    💡 Esta aplicación utiliza SymPy para el cálculo simbólico y Plotly para visualización interactiva
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()