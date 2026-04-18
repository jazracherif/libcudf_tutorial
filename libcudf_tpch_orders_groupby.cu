// libcudf_tpch_orders_groupby.cu
// Loads a TPC-H Orders table into cudf and performs a groupby + sum reduction.
//
// Usage:
//   ./libcudf_tpch_orders_groupby                          # use inline Arrow batch (TPC-H Orders)
//   ./libcudf_tpch_orders_groupby data/orders.parquet      # read from Parquet file
//
// When reading from Parquet the groupby runs on:
//   o_orderstatus  (utf8)    as key
//   o_totalprice   (float64) as value

#include <cudf/interop.hpp>                     // cudf::from_arrow (C Data Interface)
#include <cudf/io/parquet.hpp>                  // cudf::io::read_parquet
#include <cudf/groupby.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <arrow/api.h>
#include <arrow/c/bridge.h>

#include <cuda_runtime.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/logging_resource_adaptor.hpp>

#include <typeinfo>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "rmm_backtrace_resource_adaptor.hpp"

// ---------------------------------------------------------------------------
// Convenience: throw on a bad arrow::Status
// ---------------------------------------------------------------------------
static void check_arrow_status(arrow::Status const& s, char const* ctx)
{
    if (!s.ok())
        throw std::runtime_error(std::string(ctx) + ": " + s.ToString());
}
#define CHK(expr) check_arrow_status((expr), #expr)

// ---------------------------------------------------------------------------
// Build a small inline Arrow RecordBatch with the TPC-H Orders schema.
// Only the two columns used by the groupby are populated here:
//   o_orderstatus  (utf8)    — key:   'F' fulfilled | 'O' open | 'P' pending
//   o_totalprice   (float64) — value: total order value in USD
// ---------------------------------------------------------------------------
static std::shared_ptr<arrow::RecordBatch> build_arrow_batch()
{
    // o_orderstatus (utf8)
    arrow::StringBuilder status_builder;
    CHK(status_builder.Append("F"));   // fulfilled
    CHK(status_builder.Append("O"));   // open
    CHK(status_builder.Append("F"));   // fulfilled
    CHK(status_builder.Append("P"));   // pending
    CHK(status_builder.Append("O"));   // open
    std::shared_ptr<arrow::Array> status_array;
    CHK(status_builder.Finish(&status_array));

    // o_totalprice (float64)
    arrow::DoubleBuilder price_builder;
    CHK(price_builder.AppendValues({173665.47, 46929.18, 193846.25, 32151.78, 121200.00}));
    std::shared_ptr<arrow::Array> price_array;
    CHK(price_builder.Finish(&price_array));

    auto schema = arrow::schema({
        arrow::field("o_orderstatus", arrow::utf8()),
        arrow::field("o_totalprice",  arrow::float64())
    });
    return arrow::RecordBatch::Make(schema, 5, {status_array, price_array});
}

// ---------------------------------------------------------------------------
// Print helpers — copy device data back to host via cudaMemcpy
// ---------------------------------------------------------------------------

// Print a cudf strings column using strings_column_view (offsets + chars)
static void print_strings_col(cudf::column_view const& col)
{
    cudf::strings_column_view sv(col);
    int n = col.size();

    // offsets: int32_t, count = n+1
    std::vector<int32_t> h_offsets(static_cast<std::size_t>(n) + 1);
    cudaMemcpy(h_offsets.data(),
               sv.offsets().head<int32_t>() + sv.offset(),
               (n + 1) * sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    int32_t total_chars = h_offsets[n] - h_offsets[0];
    std::vector<char> h_chars(static_cast<std::size_t>(total_chars));
    cudaMemcpy(h_chars.data(),
               sv.chars_begin(rmm::cuda_stream_default),
               total_chars,
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        int32_t start = h_offsets[i]   - h_offsets[0];
        int32_t end   = h_offsets[i+1] - h_offsets[0];
        std::cout << std::string(h_chars.data() + start, end - start) << "\n";
    }
}

// Print an int32_t column
static void print_int32_col(cudf::column_view const& col)
{
    int n = col.size();
    std::vector<int32_t> h(static_cast<std::size_t>(n));
    cudaMemcpy(h.data(), col.begin<int32_t>(), n * sizeof(int32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
        std::cout << h[i] << "\n";
}

// Print an int64_t column
static void print_int64_col(cudf::column_view const& col)
{
    int n = col.size();
    std::vector<int64_t> h(static_cast<std::size_t>(n));
    cudaMemcpy(h.data(), col.begin<int64_t>(), n * sizeof(int64_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
        std::cout << h[i] << "\n";
}

// Print a float32 column
static void print_float32_col(cudf::column_view const& col)
{
    int n = col.size();
    std::vector<float> h(static_cast<std::size_t>(n));
    cudaMemcpy(h.data(), col.begin<float>(), n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
        std::cout << h[i] << "\n";
}

// Print a float64 column
static void print_float64_col(cudf::column_view const& col)
{
    int n = col.size();
    std::vector<double> h(static_cast<std::size_t>(n));
    cudaMemcpy(h.data(), col.begin<double>(), n * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
        std::cout << h[i] << "\n";
}

// Print a bool column (stored as uint8_t)
static void print_bool_col(cudf::column_view const& col)
{
    int n = col.size();
    std::vector<uint8_t> h(static_cast<std::size_t>(n));
    cudaMemcpy(h.data(), col.begin<uint8_t>(), n * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i)
        std::cout << (h[i] ? "true" : "false") << "\n";
}

// Dispatch print based on column type_id
static void print_column(cudf::column_view const& col)
{
    using cudf::type_id;
    switch (col.type().id()) {
        case type_id::INT8:
        case type_id::INT16:
        case type_id::INT32:  print_int32_col(col);  break;
        case type_id::INT64:
        case type_id::TIMESTAMP_MILLISECONDS:
        case type_id::TIMESTAMP_MICROSECONDS:
        case type_id::TIMESTAMP_NANOSECONDS:
        case type_id::TIMESTAMP_SECONDS:      print_int64_col(col);  break;
        case type_id::FLOAT32:                print_float32_col(col); break;
        case type_id::FLOAT64:                print_float64_col(col); break;
        case type_id::BOOL8:                  print_bool_col(col);   break;
        case type_id::STRING:                 print_strings_col(col); break;
        default:
            std::cout << "(unsupported type: " << static_cast<int>(col.type().id()) << ")\n";
            break;
    }
}

// Print all columns of a table_view, with optional column names
static void print_table(cudf::table_view const& tv,
                        std::vector<std::string> const& col_names = {})
{
    int ncols = tv.num_columns();
    int nrows = tv.num_rows();
    std::cout << "=== DataFrame: " << ncols << " columns, " << nrows << " rows ===\n";
    for (int c = 0; c < ncols; ++c) {
        std::string label = (c < static_cast<int>(col_names.size()))
                            ? col_names[c]
                            : ("col[" + std::to_string(c) + "]");
        std::cout << "--- " << label << " ---\n";
        print_column(tv.column(c));
    }
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// Table source descriptor
// ---------------------------------------------------------------------------
struct TableSource {
    std::unique_ptr<cudf::table> table;
    int key_col;
    int value_col;
};

// ---------------------------------------------------------------------------
// Source 1: build an inline Arrow batch and import into cudf
// ---------------------------------------------------------------------------
static TableSource from_inline_arrow()
{
    auto batch = build_arrow_batch();
    std::cout << "=== Input Arrow RecordBatch ===\n"
              << batch->ToString() << "\n";

    ArrowSchema c_schema{};
    ArrowArray  c_array{};
    CHK(arrow::ExportRecordBatch(*batch, &c_array, &c_schema));

    return {
        cudf::from_arrow(&c_schema, &c_array),
        /*key_col=*/   0,   // "o_orderstatus"
        /*value_col=*/ 1    // "o_totalprice"
    };
}

// ---------------------------------------------------------------------------
// Source 2: read a Parquet file into cudf
// Schema expected (from make_tpch_orders.py — TPC-H Orders):
//   0: o_orderkey      int64
//   1: o_custkey       int64
//   2: o_orderstatus   utf8        <-- groupby key
//   3: o_totalprice    float64     <-- sum value
//   4: o_orderdate     date32
//   5: o_orderpriority utf8
//   6: o_clerk         utf8
//   7: o_shippriority  int32
//   8: o_comment       utf8
// ---------------------------------------------------------------------------
static int find_column(cudf::io::table_with_metadata const& result,
                       std::string const& name)
{
    auto const& cols = result.metadata.schema_info;
    for (int i = 0; i < static_cast<int>(cols.size()); ++i)
        if (cols[i].name == name) return i;
    throw std::runtime_error("Column not found in Parquet schema: " + name);
}

static TableSource from_parquet(std::string const& path,
                                std::string const& key_name   = "o_orderstatus",
                                std::string const& value_name = "o_totalprice")
{
    auto source  = cudf::io::source_info(path);
    auto options = cudf::io::parquet_reader_options::builder(source).build();
    auto result  = cudf::io::read_parquet(options);

    int key_col   = find_column(result, key_name);
    int value_col = find_column(result, value_name);

    std::cout << "=== Loaded Parquet: " << path << " ("
              << result.tbl->num_columns() << " cols, "
              << result.tbl->view().num_rows() << " rows) ===\n"
              << "    key_col=\""   << key_name   << "\" [" << key_col   << "]\n"
              << "    value_col=\"" << value_name << "\" [" << value_col << "]\n";

    return {
        std::move(result.tbl),
        key_col,
        value_col
    };
}

// ---------------------------------------------------------------------------
// Run groupby + sum and print results
// ---------------------------------------------------------------------------
static void run_groupby_sum(TableSource const& src)
{
    cudf::table_view tv = src.table->view();

    // print_table(tv);

    cudf::groupby::groupby gb(cudf::table_view{{tv.column(src.key_col)}});

    std::vector<cudf::groupby::aggregation_request> requests;
    {
        cudf::groupby::aggregation_request req;
        req.values = tv.column(src.value_col);
        req.aggregations.push_back(
            cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        requests.push_back(std::move(req));
    }

    auto [grouped_keys, agg_results] = gb.aggregate(requests);

//     std::cout << "=== groupby(o_orderstatus).sum(o_totalprice) ===\n";
//     std::cout << "o_orderstatus:\n";
//     print_strings_col(grouped_keys->view().column(0));
//     std::cout << "sum(o_totalprice):\n";
//     print_float64_col(agg_results[0].results[0]->view());
}

// ---------------------------------------------------------------------------
// RAII guard that installs two RMM instrumentation adaptors and restores the
// original MR on destruction.  Construct before any cudf/RMM call.
//
//   Stack when enabled:
//     [cudf/RMM code]
//         → backtrace_resource_adaptor  (prints demangled call stack to stdout)
//         → logging_resource_adaptor    (writes CSV to rmm_alloc_log.csv)
//         → default pool MR             (actual GPU allocation)
// ---------------------------------------------------------------------------
struct RmmInstrumentationGuard {
    explicit RmmInstrumentationGuard(bool enable,
                                     std::string const& log_file = "rmm_alloc_log.csv")
        : enabled_(enable)
    {
        if (!enabled_) return;

        original_mr_ = rmm::mr::get_current_device_resource();
        log_mr_ = std::make_unique<
            rmm::mr::logging_resource_adaptor<rmm::mr::device_memory_resource>>(
                original_mr_, log_file);
        bt_mr_ = std::make_unique<backtrace_resource_adaptor>(log_mr_.get());
        rmm::mr::set_current_device_resource(bt_mr_.get());

        std::cout << "[RMM instrumentation enabled]  log → " << log_file << "\n\n";
    }

    ~RmmInstrumentationGuard()
    {
        if (!enabled_) return;
        rmm::mr::set_current_device_resource(original_mr_);
    }

    // Non-copyable, non-movable — the adaptors hold raw pointers to each other.
    RmmInstrumentationGuard(RmmInstrumentationGuard const&)            = delete;
    RmmInstrumentationGuard& operator=(RmmInstrumentationGuard const&) = delete;

private:
    bool enabled_{false};
    rmm::mr::device_memory_resource* original_mr_{nullptr};
    std::unique_ptr<rmm::mr::logging_resource_adaptor<rmm::mr::device_memory_resource>> log_mr_;
    std::unique_ptr<backtrace_resource_adaptor> bt_mr_;
};

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Options:
    //   --input <path>   Parquet file to read (default: use inline Arrow data)
    //   --rmm-trace      Print a demangled CPU call stack for every RMM alloc/dealloc
    //                    Also enabled by setting RMM_INSTRUMENT=1 in the environment.
    bool        trace    = static_cast<bool>(std::getenv("RMM_INSTRUMENT"));
    std::string parquet_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--rmm-trace") {
            trace = true;
        } else if ((arg == "--input" || arg == "-i") && i + 1 < argc) {
            parquet_path = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << "\n"
                      << "Usage: " << argv[0]
                      << " [--input <parquet>] [--rmm-trace]\n";
            return 1;
        }
    }

    RmmInstrumentationGuard rmm_guard{trace};

    TableSource src = !parquet_path.empty()
        ? from_parquet(parquet_path)
        : from_inline_arrow();

    run_groupby_sum(src);
    return 0;
}

