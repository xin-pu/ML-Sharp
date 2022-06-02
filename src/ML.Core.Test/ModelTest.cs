using System.IO;
using FluentAssertions;
using ML.Core.Data;
using ML.Core.Data.Loader;
using ML.Core.Models;
using ML.Core.Test.DataStructs;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Test
{
    public class ModelTest : AbstractTest
    {
        private readonly string dataFolder = @"..\..\..\..\..\data";

        public ModelTest(ITestOutputHelper testOutputHelper)
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void TestCreateModel()
        {
            var model = new MultipleLinearRegression<IrisData>();
            print(model);
            model.Weights.Should().BeNull();
        }


        [Fact]
        public void TestModel()
        {
            var path = Path.Combine(dataFolder, "iris-train.txt");
            var data = TextLoader<IrisData>.LoadDataSet(path, splitChar: '\t');

            var model = new MultipleLinearRegression<IrisData>();
            model.PipelineDataSet(data);
            print(model);
            model.Weights.Should().NotBeNull();

            var iEnumerator = data.GetEnumerator();
            if (iEnumerator.MoveNext() && iEnumerator.Current is Dataset<IrisData> batch)
            {
                var feature = batch.ToDatasetNDarray().Feature;
                model.CallGraph(feature);
                var predict = model.Call(feature);
                print(predict);
            }
        }
    }
}