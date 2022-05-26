using System;
using System.Collections.Generic;

namespace ML.Core.Data
{
    [AttributeUsage(AttributeTargets.Property | AttributeTargets.Field)]
    public sealed class LoadColumnAttribute : Attribute
    {
        internal List<Range> Sources;

        /// <summary>Maps member to specific field in text file.</summary>
        /// <param name="fieldIndex">The index of the field in the text file.</param>
        public LoadColumnAttribute(int fieldIndex)
        {
            Sources = new List<Range> {new Range(fieldIndex)};
        }

        /// <summary>Maps member to range of fields in text file.</summary>
        /// <param name="start">The starting field index, for the range.</param>
        /// <param name="end">The ending field index, for the range.</param>
        public LoadColumnAttribute(int start, int end)
        {
            Sources = new List<Range> {new Range(start, end)};
        }

        /// <summary>Maps member to set of fields in text file.</summary>
        /// <param name="columnIndexes">Distinct text file field indices to load as part of this column.</param>
        public LoadColumnAttribute(int[] columnIndexes)
        {
            Sources = new List<Range>();
            foreach (var columnIndex in columnIndexes)
                Sources.Add(new Range(columnIndex));
        }
    }
}