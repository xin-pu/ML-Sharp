namespace ML.Core.Data.Test
{
    public class IrisData
    {
        [LoadColumn(0)] public float Label;

        [LoadColumn(3)] public float PetalLength;

        [LoadColumn(4)] public float PetalWidth;

        [LoadColumn(1)] public float SepalLength;

        [LoadColumn(2)] public float SepalWidth;
    }
}