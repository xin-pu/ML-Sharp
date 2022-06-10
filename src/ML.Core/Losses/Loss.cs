using System;
using System.ComponentModel;
using System.Linq;
using System.Text;
using AutoDiff;
using FluentAssertions;
using GalaSoft.MvvmLight;
using ML.Utility;
using Numpy;

namespace ML.Core.Losses
{
    public abstract class Loss : ViewModelBase, IRecorder
    {
        private double _lamdba = 1E-4;
        private Regularization _regularization = Regularization.None;

        /// <summary>
        ///     损失抽象类
        /// </summary>
        /// <param name="lamdba"></param>
        /// <param name="regularization">regularization type</param>
        protected Loss()
        {
        }


        /// <summary>
        ///     正则化，惩罚参数
        /// </summary>
        [Category("Configuration")]
        public double Lamdba
        {
            get => _lamdba;
            set => Set(ref _lamdba, value);
        }

        /// <summary>
        ///     正则模式，约束模式
        /// </summary>
        [Category("Configuration")]
        public Regularization Regularization
        {
            get => _regularization;
            set => Set(ref _regularization, value);
        }

        [Category("Tag")] public abstract string Describe { get; }

        public Action<double> ReportToRecorder { get; set; }

        [Category("Tag")] public string Name => GetType().Name;

        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine($"---{Name}---");
            return str.ToString();
        }

        #region Calculate loss

        /// <summary>
        ///     直接计算损失
        /// </summary>
        /// <param name="y_pred">[batch size, ... ]</param>
        /// <param name="y_true">[batch size, ... ]</param>
        /// <returns></returns>
        public virtual double GetLoss(NDarray y_pred, NDarray y_true)
        {
            y_pred.size.Should().Be(y_true.size, "size of pred and ture should be same.");
            var y_pred_reshape = np.reshape(y_pred, y_true.shape);

            checkLabels(y_true);
            var loss = calculateLoss(y_pred_reshape, y_true);
            ReportToRecorder?.Invoke(loss);

            return loss;
        }

        #endregion

        #region Loss- AutoDiff

        /// <summary>
        ///     获取损失
        /// </summary>
        /// <param name="y_pred">模型预测</param>
        /// <param name="y_true">真实Y</param>
        /// <param name="variables">模型变量</param>
        /// <returns></returns>
        public virtual Term GetLossTerm(TermMatrix y_pred, NDarray y_true, Variable[] variables)
        {
            variables.Should().NotBeNullOrEmpty("Variables contains null value.");
            y_true.shape[0].Should().Be(y_pred.Height, "Batch size should be same.");
            y_true.shape[1].Should().Be(y_pred.Width, "Pred size should be same.");

            checkLabels(y_true);

            var basicLoss = getModelLoss(y_pred, y_true);
            var regularizationLoss = getRegularizationLoss(variables, Regularization, Lamdba);
            var totalLoss = basicLoss + regularizationLoss;
            return totalLoss;
        }


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
                case Regularization.Lasso_L1:
                    return lamdba * getLassoLoss(variables);
                case Regularization.Ridge_L2:
                    return lamdba * getRidgeLoss(variables) / 2;
                case Regularization.ElasticNet_LP:
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

        #endregion

        #region Internal

        internal abstract void checkLabels(NDarray y_true);
        internal abstract double calculateLoss(NDarray y_pred, NDarray y_true);
        internal abstract Term getModelLoss(TermMatrix y_pred, NDarray y_true);

        #endregion
    }


    public enum LabelType
    {
        Logits,
        Probability
    }
}