using System.Collections.Generic;
using System.IO;
using System.Linq;
using FluentAssertions;
using ML.Core.Data.Loader;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Data.Test
{
    public class DataSetTest
    {
        private readonly ITestOutputHelper _testOutputHelper;

        private readonly string dataFolder = @"..\..\..\..\..\data";

        public DataSetTest(ITestOutputHelper testOutputHelper)
        {
            _testOutputHelper = testOutputHelper;
        }

        internal void print(object obj)
        {
            _testOutputHelper.WriteLine(obj.ToString());
        }

        public DataSet<IrisData> GetIrisDataSet(int count)
        {
            var iris = new List<IrisData>();
            Enumerable.Range(0, count).ToList()
                .ForEach(_ => iris.Add(IrisData.RandomIris()));
            var dataset = new DataSet<IrisData>(iris);
            return dataset;
        }

        [Fact]
        public void DataSetCreate()
        {
            var dataset = GetIrisDataSet(1);
            print(dataset);
        }

        [Fact]
        public void DataSetSplit()
        {
            var dataset = GetIrisDataSet(10);
            var res = dataset.Split(0.8);
            print(res.Item1);
            print(res.Item2);
        }

        [Fact]
        public void DataSetOrder()
        {
            var dataset = GetIrisDataSet(5);
            var orderDataSet = dataset.Orderby(c => c.SepalLength);
            print(orderDataSet);
        }

        [Fact]
        public void DataSetShuffle()
        {
            var dataset = GetIrisDataSet(10);
            var shuffleDataSet = dataset.Shuffle();
            print(dataset);
            print(shuffleDataSet);
        }


        [Fact]
        public void DataSetClone()
        {
            var dataset = GetIrisDataSet(1);
            var cloneDataSet = dataset.Clone();
            cloneDataSet.Should().NotBe(dataset);
            print(dataset);
            print(cloneDataSet);
        }

        [Fact]
        public void DataSetConcat()
        {
            var dataset1 = GetIrisDataSet(1);
            var dataset2 = GetIrisDataSet(1);
            var concatOne = dataset1.Concat(dataset2);
            print(concatOne);
        }


        [Fact]
        public void DataSetRepeat()
        {
            var dataset1 = GetIrisDataSet(1);
            var repeatOne = dataset1.Repeat(5);
            print(repeatOne);
        }

        [Fact]
        public void DataSetLoadTest()
        {
            var path = Path.Combine(dataFolder, "iris-train.txt");
            var dataset = TextLoader.LoadDataSet<IrisData>(path, splitChar: '\t');
            print(dataset);
        }
    }
}