namespace ML.Core.Transform
{
    public abstract class Kernel : Transformer
    {
        protected Kernel(KernelType kernelType = KernelType.Gauss)
        {
            KernelType = kernelType;
        }

        public KernelType KernelType { protected set; get; }

        public override bool IsKernel => true;
    }

    public enum KernelType
    {
        Lapras,
        Poly,
        Gauss,
        Sigmoid
    }
}