using ML.Core.Transform;

namespace ML.Core.Models
{
    public class PolynomialRegression : MultipleLinearRegression
    {
        /// <summary>
        ///     一元多项式回归
        ///     y=α + β1*x + β2*x^2 + ... + βn*x^n
        /// </summary>
        public PolynomialRegression(int degree)
        {
            Degree = degree;
            Transformer = new Polynomial(degree);
        }

        public int Degree { protected set; get; }


        public override string Description => "多项式回归\r\n y=α + β1*x + β2*x^2 + ... + βn*x^n";
    }
}