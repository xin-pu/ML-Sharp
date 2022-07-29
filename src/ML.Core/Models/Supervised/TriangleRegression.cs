using ML.Core.Transform;

namespace ML.Core.Models
{
    public class TriangleRegression : MultipleLinearRegression
    {
        private int _degree;

        /// <summary>
        ///     一元多项式回归
        ///     y=α + β1*x + β2*x^2 + ... + βn*x^n
        /// </summary>
        public TriangleRegression()
        {
            Degree = 1;
        }

        /// <summary>
        ///     一元多项式回归
        ///     y=α + β1*x + β2*x^2 + ... + βn*x^n
        /// </summary>
        public TriangleRegression(int degree)
        {
            Degree = degree;
        }

        public int Degree
        {
            set
            {
                Set(ref _degree, value);
                UpdateTransform();
            }
            get => _degree;
        }

        public override string Description => "多项式回归\r\n y=α + β1*x + β2*x^2 + ... + βn*x^n";


        private void UpdateTransform()
        {
            Transformer = new TrianglePoly(Degree);
        }
    }
}