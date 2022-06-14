using System.IO;
using System.Linq;
using System.Text;
using AutoDiff;
using FluentAssertions;
using GalaSoft.MvvmLight;
using ML.Core.Data;
using ML.Core.Transform;
using ML.Utility;
using Numpy;
using YAXLib;
using YAXLib.Attributes;

namespace ML.Core.Models
{
    /// <summary>
    ///     用于梯度下降法的基本模型
    ///     需要定义损失函数
    ///     需要定义优化器
    ///     需要定义转换器
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public abstract class ModelGD : ViewModelBase, IModelGD
    {
        private InitialWeigts _initialWeights = InitialWeigts.False;
        private Transformer _transformer;
        private Variable[] _variables;

        private WeightInitial _weightInitial;
        private NDarray _weights;

        /// <summary>
        /// </summary>
        protected ModelGD()
        {
        }

        public abstract string Description { get; }

        public string Name => GetType().Name;

        public Transformer Transformer
        {
            get => _transformer;
            protected set => Set(ref _transformer, value);
        }

        public InitialWeigts InitialWeights
        {
            get => _initialWeights;
            protected set => Set(ref _initialWeights, value);
        }

        public WeightInitial WeightInitial
        {
            get => _weightInitial;
            set => Set(ref _weightInitial, value);
        }

        public string WeightFile => $"{Name}.txt";


        [YAXDontSerialize]
        public Variable[] Variables
        {
            get => _variables;
            set => Set(ref _variables, value);
        }

        /// <summary>
        ///     [Labels,Features]
        /// </summary>
        [YAXDontSerialize]
        public NDarray Weights
        {
            get => _weights;
            set => Set(ref _weights, value);
        }


        /// <summary>
        ///     1.通过传入数据集=>模型变换，确认参数数量
        /// </summary>
        /// <param name="dataset"></param>
        public void PipelineDataSet(Dataset<DataView> dataset)
        {
            var dataview = dataset.ToDatasetNDarray();
            var transformNDarray = Transformer.Call(dataview.Feature);

            var featureCount = transformNDarray.shape[1];
            var labelCount = dataview.Label.shape[1];
            var enumerable = Enumerable.Range(0, featureCount * labelCount);

            Variables = enumerable
                .Select(_ => new Variable())
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

        public double[] GetWeightArray()
        {
            return Weights?.GetData<double>();
        }

        /// <summary>
        /// </summary>
        /// <param name="features">[batch size, d1, d2, ... ]</param>
        /// <returns></returns>
        public abstract TermMatrix CallGraph(NDarray features);


        /// <summary>
        ///     call by X*W
        /// </summary>
        /// <param name="features">[batch size, ... ]</param>
        /// <returns>[batch size, labels]</returns>
        public abstract NDarray Call(NDarray features);

        public NDarray Call(DataView data)
        {
            var datas = new Dataset<DataView>(new[] {data});
            return Call(datas.ToDatasetNDarray().Feature)[0];
        }

        public void Save(string filename)
        {
            np.savetxt(WeightFile, Weights);
            using var stream = File.Open(filename, FileMode.Create, FileAccess.Write, FileShare.Read);
            var serializer = new YAXSerializer(typeof(ModelGD));
            using var textWriter = new StreamWriter(stream);
            serializer.Serialize(this, textWriter);
            textWriter.Flush();
        }

        public static IModelGD Load(string filename)
        {
            using var stream = File.Open(filename, FileMode.Open, FileAccess.Read, FileShare.Read);
            var serializer = new YAXSerializer(typeof(ModelGD));
            using var textWriter = new StreamReader(stream);
            var model = (IModelGD) serializer.Deserialize(textWriter);
            model.Weights = np.loadtxt(model.WeightFile);
            return model;
        }

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
                str.AppendLine($"Weight:\r{Weights}");
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