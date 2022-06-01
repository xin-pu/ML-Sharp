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
        private double _lamdba;
        private Regularization _regularization;

        /// <summary>
        ///     算是抽象类
        /// </summary>
        /// <param name="regularization1"></param>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        protected Loss(double lamdba = 1E-4,
            Regularization regularization = Regularization.None)
        {
            Lamdba = lamdba;
            Regularization = regularization;
        }

        public string Name => GetType().Name;

        public Term LossTerm { protected set; get; }

        public Variable[] Variables { protected set; get; }

        /// <summary>
        ///     正则化，惩罚参数
        /// </summary>
        public double Lamdba
        {
            get => _lamdba;
            set => SetProperty(ref _lamdba, value);
        }

        /// <summary>
        ///     正则模式，约束模式
        /// </summary>
        public Regularization Regularization
        {
            get => _regularization;
            set => SetProperty(ref _regularization, value);
        }

        /// <summary>
        ///     赋模型变量
        /// </summary>
        /// <param name="variables"></param>
        public void Complie(Variable[] variables)
        {
            Variables = variables;
        }

        /// <summary>
        ///     获取损失
        /// </summary>
        /// <param name="y_pred"></param>
        /// <param name="y_true"></param>
        /// <returns></returns>
        public Term Call(Term[] y_pred, NDarray y_true)
        {
            Variables.Should().NotBeNullOrEmpty("Variables contains null value.");
            y_true.shape[1].Should().Be(1, "Pred one result");
            y_true.shape[0].Should().Be(y_pred.Length, "Batch size should be same.");

            var basicLoss = getModelLoss(y_pred, y_true.GetData<double>());
            var regularizationLoss = getRegularizationLoss(Variables, Regularization, Lamdba);
            var totalLoss = basicLoss + regularizationLoss;
            return totalLoss;
        }

        /// <summary>
        ///     获取损失
        /// </summary>
        /// <param name="y_pred"></param>
        /// <param name="y_true"></param>
        /// <returns>用于计算梯度的损失关系</returns>
        public Term Call(Term[] y_pred, double[] y_true)
        {
            var y_true_array = np.expand_dims(np.array(y_true), 0);
            return Call(y_pred, y_true_array);
        }

        public double Call(NDarray y_pred, NDarray y_true)
        {
            y_pred.size.Should().Be(y_true.size, "size of pred and ture should be same.");
            var y_pred_reshape = np.reshape(y_pred, y_true.shape);
            return CalculateLLoss(y_pred_reshape, y_true);
        }

        internal abstract double CalculateLLoss(NDarray y_pred, NDarray y_true);


        /// <summary>
        ///     获取具体模型损失，由各个类型的模型自定义
        /// </summary>
        /// <param name="w"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        internal abstract Term getModelLoss(Term[] y_pred, double[] y_true);

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
                case Regularization.None:
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

    public enum Initialization
    {
        Zero,
        Rand,
        Randn
    }
}