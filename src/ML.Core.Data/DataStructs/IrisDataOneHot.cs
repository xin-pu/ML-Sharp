using Numpy;

namespace ML.Core.Data.DataStructs
{
    [Serializable]
    public class IrisDataOneHot : IrisData
    {
        /// <summary>
        ///     Cons
        /// </summary>
        public IrisDataOneHot()
        {
        }

        public override NDarray GetLabelArray()
        {
            var array = new double[3];
            array[Label] = 1;
            return np.array(array);
        }


        public override string ToString()
        {
            return
                $"Label:{Label}\t" +
                $"SepalLength:{SepalLength:F2}\tSepalWidth:{SepalWidth:F2}\t" +
                $"PetalLength:{PetalLength:F2}\tPetalWidth:{PetalWidth:F2}";
        }
    }
}