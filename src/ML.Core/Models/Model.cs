using System.Linq;
using System.Text;
using AutoDiff;
using ML.Core.Data;
using ML.Core.Transform;
using MvvmCross.ViewModels;
using Numpy;

namespace ML.Core.Models
{
    /// <summary>
    ///     用于梯度下降法的基本模型
    ///     需要定义损失函数
    ///     需要定义优化器
    ///     需要定义转换器
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class Model<T> : MvxViewModel
        where T : DataView
    {
        private InitialWeigts _initialWeights = InitialWeigts.False;
        private Variable[] _variables;
        private double[] _weights;

        public string Name => GetType().Name;

        public abstract Transformer Transformer { set; get; }

        public Variable[] Variables
        {
            get => _variables;
            protected set => SetProperty(ref _variables, value);
        }

        public double[] Weights
        {
            get => _weights;
            protected set => SetProperty(ref _weights, value);
        }

        public InitialWeigts InitialWeights
        {
            get => _initialWeights;
            protected set => SetProperty(ref _initialWeights, value);
        }

        internal NDarray WeightNDarray => np.expand_dims(np.array(Weights), -1);

        /// <summary>
        ///     1.通过传入数据集=>模型变换，确认参数数量
        /// </summary>
        /// <param name="dataset"></param>
        public void PipelineDataSet(Dataset<T> dataset, WeightInitial weightInitial = WeightInitial.Rand)
        {
            var feature = dataset.ToDatasetNDarray().Feature;
            var transformNDarray = Transformer.Call(feature);

            var featureCount = transformNDarray.shape[1];
            var enumerable = Enumerable.Range(0, featureCount);

            Variables = enumerable
                .Select(i => new Variable())
                .ToArray();


            switch (weightInitial)
            {
                case WeightInitial.One:
                    Weights = np.ones(featureCount).GetData<double>();
                    break;
                case WeightInitial.Zero:
                    Weights = np.zeros(featureCount).GetData<double>();
                    break;
                case WeightInitial.Rand:
                default:
                    Weights = np.random.rand(featureCount).GetData<double>();
                    break;
            }

            InitialWeights = InitialWeigts.True;
        }

        /// <summary>
        /// </summary>
        /// <param name="x">[batch size, d1, d2, ... ]</param>
        /// <returns></returns>
        public abstract Term[] CallGraph(NDarray x);

        public abstract NDarray Call(NDarray x);

        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine(Name);
            str.AppendLine($"{Transformer}");
            if (InitialWeights == InitialWeigts.True)
            {
                str.AppendLine($"ParaCount:\t{Variables.Length}");
                str.AppendLine($"ParaData:\t{WeightNDarray}");
            }

            return str.ToString();
        }
    }

    public enum WeightInitial
    {
        Rand,
        One,
        Zero
    }

    public enum InitialWeigts
    {
        True,
        False
    }
}