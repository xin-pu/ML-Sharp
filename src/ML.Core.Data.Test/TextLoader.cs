using System.Collections.Generic;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace ML.Core.Data.Test
{
    public class TextLoader
    {
        private readonly ITestOutputHelper _testOutputHelper;

        public TextLoader(ITestOutputHelper testOutputHelper)
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
                .ForEach(i => iris.Add(IrisData.RandomIris()));
            var dataset = new DataSet<IrisData>(iris);
            return dataset;
        }

        [Fact]
        public void DataSetCreate()
        {
            var dataset = GetIrisDataSet(10);
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
            var dataset = GetIrisDataSet(10);
            var orderDataSet = dataset.Orderby(c => c.SepalLength);
            print(orderDataSet);
        }
    }
}