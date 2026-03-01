from compressor import compress_document

if __name__ == "__main__":
    final_summary = compress_document()

    with open("results/compressed_output.txt", "w") as f:
        f.write(final_summary)

    print("\nCompression complete.")
    print("Saved to results/compressed_output.txt")