using System.Linq;
using System.Text;
using AutoDiff;
using FluentAssertions;
using ML.Core.Data;
using ML.Core.Transform;
using ML.Utility;
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
    public abstract class ModelGD<T> : MvxViewModel, IModelGD<T>
        where T : DataView
    {
        private InitialWeigts _initialWeights = InitialWeigts.False;
        private Variable[] _variables;

        private WeightInitial _weightInitial;
        private NDarray _weights;

        public string Name => GetType().Name;

        public Transformer Transformer { set; get; }


        public InitialWeigts InitialWeights
        {
            get => _initialWeights;
            protected set => SetProperty(ref _initialWeights, value);
        }

        public abstract string Description { get; }

        public WeightInitial WeightInitial
        {
            get => _weightInitial;
            set => SetProperty(ref _weightInitial, value);
        }

        public Variable[] Variables
        {
            get => _variables;
            set => SetProperty(ref _variables, value);
        }

        /// <summary>
        ///     [Labels,Features]
        /// </summary>
        public NDarray Weights
        {
            get => _weights;
            set => SetProperty(ref _weights, value);
        }


        /// <summary>
        ///     1.通过传入数据集=>模型变换，确认参数数量
        /// </summary>
        /// <param name="dataset"></param>
        public void PipelineDataSet(Dataset<T> dataset)
        {
            var dataview = dataset.ToDatasetNDarray();
            var transformNDarray = Transformer.Call(dataview.Feature);

            var featureCount = transformNDarray.shape[1];
            var labelCount = dataview.Label.shape[1];
            var enumerable = Enumerable.Range(0, featureCount);

            Variables = enumerable
                .Select(i => new Variable())
                .ToArray();


            switch (WeightInitial)
            {
                case WeightInitial.One:
                    Weights = np.ones(labelCount, featureCount);
                    break;
                case WeightInitial.Zero:
                    Weights = np.zeros(labelCount, featureCount);
                    break;
                case WeightInitial.Rand:
                default:
                    Weights = np.random.rand(labelCount, featureCount);
                    break;
            }

            InitialWeights = InitialWeigts.True;
        }

        /// <summary>
        /// </summary>
        /// <param name="features">[batch size, d1, d2, ... ]</param>
        /// <returns></returns>
        public abstract TermMatrix CallGraph(NDarray features);


        public double[] GetWeightArray()
        {
            return Weights?.GetData<double>();
        }

        /// <summary>
        ///     call by X*W
        /// </summary>
        /// <param name="features">[batch size, ... ]</param>
        /// <returns>[batch size, labels]</returns>
        public abstract NDarray Call(NDarray features);

        public void UpdateWeights(NDarray weightNDarray)
        {
            weightNDarray.shape.Should().BeEquivalentTo(Weights.shape, "Weigts shape should keep.");

            Weights = weightNDarray;
        }

        public override string ToString()
        {
            var str = new StringBuilder();
            str.AppendLine(Name);
            str.AppendLine(Description);
            str.AppendLine($"{Transformer}");
            if (InitialWeights == InitialWeigts.True)
            {
                str.AppendLine($"ParaCount:\t{Variables.Length}");
                str.AppendLine($"Weight:\t{Weights}");
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