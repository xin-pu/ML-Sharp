using System.Linq;
using AutoDiff;
using FluentAssertions;
using Numpy;

namespace ML.Core.Losses
{
    public class LeastSquares : Loss
    {
        /// <summary>
        ///     J(la)= 0.5*sigma((y-yp)^2)
        /// </summary>
        public LeastSquares(Regularization regularization)
            : base(regularization)
        {
        }

        internal override Term getModelLoss(Variable[] w, NDarray x, NDarray y)
        {
            y.shape[1].Should().Be(1, "Pred one result");
            x.shape[0].Should().Be(y.shape[0], "Batch size should be same.");
            w.Length.Should().Be(x.shape[1], "Variables length should math with x's features.");

            var y_delta = term.matmul(w, x, y);
            var allSquares = y_delta
                .Select(i => TermBuilder.Power(i, 2))
                .ToArray();

            return 0.5 * TermBuilder.Sum(allSquares) / x.shape[0];
        }
    }
}