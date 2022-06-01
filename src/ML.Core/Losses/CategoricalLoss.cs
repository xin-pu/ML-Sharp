using System.Linq;
using AutoDiff;
using Numpy;

namespace ML.Core.Losses
{
    public abstract class CategoricalLoss : Loss
    {
        private LabelType _labelType;

        /// <summary>
        ///     分类损失抽象类
        /// </summary>
        /// <param name="regularization1"></param>
        /// <param name="lamdba"></param>
        /// <param name="regularization"></param>
        protected CategoricalLoss(
            LabelType labelType = LabelType.Probability,
            double lamdba = 1E-4,
            Regularization regularization = Regularization.None)
            : base(lamdba, regularization)
        {
            LabelType = labelType;
        }

        public LabelType LabelType
        {
            get => _labelType;
            set => SetProperty(ref _labelType, value);
        }

        /// <summary>
        ///     获取损失
        /// </summary>
        /// <param name="y_pred">[batch size, ... ]</param>
        /// <param name="y_true">[batch size, ... ]</param>
        /// <returns></returns>
        public override Term GetLossTerm(Term[] y_pred, NDarray y_true, Variable[] variables)
        {
            var tems =
                LabelType == LabelType.Probability
                    ? y_pred
                    : y_pred.Select(convertProbabilityTerm).ToArray();

            return base.GetLossTerm(tems, y_true, variables);
        }


        /// <summary>
        ///     直接计算损失
        /// </summary>
        /// <param name="y_pred">模型预测</param>
        /// <param name="y_true">真实Y</param>
        /// <param name="variables">模型变量</param>
        /// <returns></returns>
        public override double GetLoss(NDarray y_pred, NDarray y_true)
        {
            var y_pred_array =
                LabelType == LabelType.Probability
                    ? y_pred
                    : convertProbabilityNDarray(y_pred);

            return base.GetLoss(y_pred_array, y_true);
        }

        #region Internal

        internal abstract Term convertProbabilityTerm(Term term);

        internal abstract NDarray convertProbabilityNDarray(NDarray value);

        #endregion
    }
}