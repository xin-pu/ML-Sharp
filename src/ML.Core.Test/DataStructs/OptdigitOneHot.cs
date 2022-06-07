using Numpy;

namespace ML.Core.Test.DataStructs
{
    public class OptdigitOneHot : OptdigitData
    {
        public override NDarray GetLabelArray()
        {
            var array = new double[10];
            array[(int) Label] = 1;
            return np.array(array);
        }
    }
}