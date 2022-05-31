namespace ML.Core.Losses
{
    /// <summary>
    ///     权重约束
    /// </summary>
    public enum Regularization
    {
        None = 0,
        L1 = 1,
        L2 = 2,
        LP = 3,
        Ridge = 2,
        Lasso = 1,
        ElasticNet = 3
    }
}