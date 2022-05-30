using System.Linq;
using System.Text;
using AutoDiff;
using FluentAssertions;
using MvvmCross.ViewModels;
using Numpy;

namespace ML.Core.Losses
{
    public abstract class Loss : MvxViewModel
    {
        protected Loss(
            Regularization regularization = Regularization.None)
        {
            Regularization = regularization;
        }

        public string Name => GetType().Name;

        public Term LossTerm { protected set; get; }

        public Variable[] Variables { protected set; get; }

        public double Lamdba { set; get; } = 0.1;

        public Regularization Regularization { protected set; get; }

        /// <summary>
        ///     Give Model's Variables to Loss
        /// </summary>
        /// <param name="variables"></param>
        public void Complie(Variable[] variables)
        {
            Variables = variables;
        }

        public (NDarray, double) Call(NDarray weights, NDarray x, NDarray y)
        {
            x.shape[0].Should().Be(y.shape[0], "batch size of X and Y should be same.");

            LossTerm = getTotalLoss(Variables, x, y);
            var points = weights.GetData<double>();
            var loss = LossTerm.Evaluate(Variables, points);
            var grad = LossTerm.Differentiate(Variables, points);
            return (grad, loss);
        }


        internal Term getTotalLoss(Variable[] w, NDarray x, NDarray y)
        {
            var basicLoss = getModelLoss(w, x, y);
            var regularizationLoss = getRegularizationLoss(w, Regularization, Lamdba);
            return basicLoss + regularizationLoss;
        }

        internal abstract Term getModelLoss(Variable[] w, NDarray x, NDarray y);

        /// <summary>
        ///     正则化约束
        /// </summary>
        /// <param name="variables"></param>
        /// <param name="regularization"></param>
        /// <param name="lamdba"></param>
        /// <returns></returns>
        internal Term getRegularizationLoss(Variable[] variables, Regularization regularization, double lamdba)
        {
            switch (regularization)
            {
                case Regularization.L1:
                    return lamdba * getLassoLoss(variables);
                case Regularization.L2:
                    return lamdba * getRidgeLoss(variables) / 2;
                case Regularization.LP:
                    return (1 - lamdba) * getLassoLoss(variables) + lamdba * getRidgeLoss(variables);
                default:
                    return new Constant(0);
            }
        }

        /// <summary>
        ///     防止过拟合 岭回归部分 L2约束
        /// </summary>
        /// <param name="w"></param>
        /// <returns></returns>
        private Term getRidgeLoss(Variable[] w)
        {
            var sum = TermBuilder.Sum(w.Select(a => TermBuilder.Power(a, 2)));
            var Ter = TermBuilder.Power(sum, 0.5);
            return Ter;
        }

        /// <summary>
        ///     防止过拟合 岭回归部分 L2约束
        /// </summary>
        /// <param name="w"></param>
        /// <returns></returns>
        private Term getLassoLoss(Variable[] w)
        {
            var abs = w.Select(i => TermBuilder.Power(TermBuilder.Power(i, 2), 0.5));
            var sum = TermBuilder.Sum(abs);
            return sum;
        }


        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine($"---{Name}---");
            return str.ToString();
        }
    }

    /// <summary>
    ///     权重约束
    /// </summary>
    public enum Regularization
    {
        None = 0,
        L1 = 1,
        L2 = 2,
        LP = 3,
        Ridge = 2,
        Lasso = 1,
        ElasticNet = 3
    }

    public enum Initialization
    {
        Zero,
        Rand,
        Randn
    }
}