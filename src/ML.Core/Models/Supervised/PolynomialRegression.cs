using ML.Core.Transform;

namespace ML.Core.Models
{
    public class PolynomialRegression : MultipleLinearRegression
    {
        private int _degree;

        public PolynomialRegression()
        {
            Degree = 1;
        }

        /// <summary>
        ///     一元多项式回归
        ///     y=α + β1*x + β2*x^2 + ... + βn*x^n
        /// </summary>
        public PolynomialRegression(int degree = 1)
        {
            Degree = degree;
        }

        public int Degree
        {
            set
            {
                SetProperty(ref _degree, value);
                UpdateTransform();
            }
            get => _degree;
        }

        public override string Description => "多项式回归\r\n y=α + β1*x + β2*x^2 + ... + βn*x^n";


        private void UpdateTransform()
        {
            Transformer = new Polynomial(Degree);
        }
    }
}