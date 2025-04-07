use std::env;
use std::fmt::Debug;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const TAG: &str = "v2.19.0";
const TF_GIT_URL: &str = "https://github.com/tensorflow/tensorflow.git";

// Environment variables for customization
const BAZEL_COPTS_ENV_VAR: &str = "TFLITEC_BAZEL_COPTS";
const PREBUILT_PATH_ENV_VAR: &str = "TFLITEC_PREBUILT_PATH";
const HEADER_DIR_ENV_VAR: &str = "TFLITEC_HEADER_DIR";

fn main() {
    // 1. Tell Cargo to re-run this build script if any of the following env variables change
    track_env_vars();

    // 2. Collect basic environment info
    let out_path = out_dir();
    let os = target_os();
    let arch = target_arch();

    // 3. Add library search path and link the library for non-iOS platforms
    //    For iOS, link a framework.
    add_link_search_and_lib(&os, &out_path);

    // 4. If building in docs.rs environment, we cannot download or clone anything;
    //    so we use a pre-packaged zip instead.
    if env::var("DOCS_RS") == Ok(String::from("1")) {
        prepare_for_docsrs();
    } else {
        let tf_src_path = out_path.join(format!("tensorflow_{}", TAG));
        let lib_output_path = lib_output_path(&os);

        // 5. If a prebuilt TFLiteC library is specified, just install it
        if let Some(prebuilt_tflitec_path) = get_target_dependent_env_var(PREBUILT_PATH_ENV_VAR) {
            install_prebuilt(&prebuilt_tflitec_path, &tf_src_path, &lib_output_path);
        } else {
            // 6. Otherwise, build from source
            check_and_set_envs();
            prepare_tensorflow_source(&tf_src_path);

            // 7. Determine the Bazel config string (like "android_arm", "ios_arm64", etc.)
            let config = compute_bazel_config(&os, &arch);
            build_tensorflow_with_bazel(
                tf_src_path.to_str().unwrap(),
                &config,
                &lib_output_path,
                &os,
            );
        }

        // 8. Generate bindings via bindgen
        generate_bindings(&tf_src_path);
    }
}

// ------------------------------------------------------------------------
// ENVIRONMENT & TARGET LOGIC
// ------------------------------------------------------------------------

/// Ensures Cargo will rerun this build script if the following environment variables change
fn track_env_vars() {
    let env_vars = [
        BAZEL_COPTS_ENV_VAR,
        PREBUILT_PATH_ENV_VAR,
        HEADER_DIR_ENV_VAR,
    ];
    for env_var in env_vars {
        println!("cargo:rerun-if-env-changed={env_var}");
        // Also consider the variant with normalized target suffix: e.g. BAZEL_COPTS_x86_64_UNKNOWN_LINUX_GNU
        if let Some(target) = normalized_target() {
            println!("cargo:rerun-if-env-changed={env_var}_{target}");
        }
    }
}

/// Returns PathBuf for the OUT_DIR
fn out_dir() -> PathBuf {
    PathBuf::from(env::var("OUT_DIR").unwrap())
}

/// Returns the target OS (e.g. "windows", "macos", "linux", "android", "ios")
fn target_os() -> String {
    env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS")
}

/// Returns the target architecture, converting aarch64->arm64, armv7->arm if building for Android, etc.
fn target_arch() -> String {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").expect("Unable to get TARGET_ARCH");
    match arch.as_str() {
        "aarch64" => "arm64".to_string(),
        "armv7" => {
            // If we're on android, rename "armv7" to "arm". Otherwise keep "armv7".
            if target_os() == "android" {
                "arm".to_string()
            } else {
                arch
            }
        }
        _ => arch,
    }
}

/// Returns the dynamic library extension (.so / .dylib / .dll)
fn dll_extension() -> &'static str {
    match target_os().as_str() {
        "macos" => "dylib",
        "windows" => "dll",
        _ => "so",
    }
}

/// Returns the library prefix (e.g. "lib" on Unix, empty string on Windows)
fn dll_prefix() -> &'static str {
    match target_os().as_str() {
        "windows" => "",
        _ => "lib",
    }
}

/// Returns a normalized string for the target triple, e.g. x86_64_unknown_linux_gnu -> X86_64_UNKNOWN_LINUX_GNU
fn normalized_target() -> Option<String> {
    env::var("TARGET")
        .ok()
        .map(|t| t.to_uppercase().replace('-', "_"))
}

/// Returns the env var in the form `VAR_{NORMALIZED_TARGET}` if it exists, otherwise returns `VAR`.
fn get_target_dependent_env_var(var: &str) -> Option<String> {
    if let Some(target) = normalized_target() {
        if let Ok(v) = env::var(format!("{var}_{target}")) {
            return Some(v);
        }
    }
    env::var(var).ok()
}

/// Sets default environment variables for Bazel if they are not already set.
fn check_and_set_envs() {
    let python_bin_path = match get_python_bin_path() {
        Some(p) => p,
        None => panic!(
            "Cannot find Python binary with required packages (numpy, importlib.util). \
             Set PYTHON_BIN_PATH or make sure 'which python3'/'python' is correct."
        ),
    };

    let defaults = [
        ("PYTHON_BIN_PATH", python_bin_path.to_str().unwrap()),
        ("USE_DEFAULT_PYTHON_LIB_PATH", "1"),
        ("TF_NEED_OPENCL", "0"),
        ("TF_CUDA_CLANG", "0"),
        ("TF_NEED_TENSORRT", "0"),
        ("TF_DOWNLOAD_CLANG", "0"),
        ("TF_NEED_MPI", "0"),
        ("TF_NEED_ROCM", "0"),
        ("TF_NEED_CUDA", "0"),
        // For Windows (though it doesn't hurt on other platforms)
        ("TF_OVERRIDE_EIGEN_STRONG_INLINE", "1"),
        ("CC_OPT_FLAGS", "-Wno-sign-compare"),
    ];

    for (k, v) in defaults {
        if env::var_os(k).is_none() {
            env::set_var(k, v);
        }
    }

    let os = target_os();
    if os == "android" {
        env::set_var("TF_SET_ANDROID_WORKSPACE", "1");
        let android_env_vars = [
            "ANDROID_NDK_HOME",
            "ANDROID_NDK_API_LEVEL",
            "ANDROID_SDK_HOME",
            "ANDROID_API_LEVEL",
            "ANDROID_BUILD_TOOLS_VERSION",
        ];
        for name in android_env_vars {
            if env::var(name).is_err() {
                panic!("{} should be set for Android build", name);
            }
        }
    } else {
        env::set_var("TF_SET_ANDROID_WORKSPACE", "0");
    }

    if os == "ios" {
        env::set_var("TF_CONFIGURE_IOS", "1");
    } else {
        env::set_var("TF_CONFIGURE_IOS", "0");
    }
}

/// Creates a config string for Bazel builds. e.g. for iOS-arm64 we do "ios_arm64", for Android-arm we do "android_arm", etc.
fn compute_bazel_config(os: &str, arch: &str) -> String {
    if os == "android" || os == "ios" || (os == "macos" && arch == "arm64") {
        format!("{}_{}", os, arch)
    } else {
        os.to_string()
    }
}

// ------------------------------------------------------------------------
// PYTHON LOGIC
// ------------------------------------------------------------------------

/// Checks if the given python bin path can import `numpy` and `importlib.util`
fn test_python_bin(python_bin_path: &str) -> bool {
    let status = std::process::Command::new(python_bin_path)
        .args(["-c", "import numpy; import importlib.util"])
        .status();
    matches!(status, Ok(s) if s.success())
}

/// Tries to find a suitable Python binary by checking PYTHON_BIN_PATH, then searching for "python3"/"python".
fn get_python_bin_path() -> Option<PathBuf> {
    if let Ok(val) = env::var("PYTHON_BIN_PATH") {
        if !test_python_bin(&val) {
            panic!(
                "The specified PYTHON_BIN_PATH '{}' failed the import test!",
                val
            );
        }
        return Some(PathBuf::from(val));
    }

    let bin = if target_os() == "windows" {
        "where"
    } else {
        "which"
    };

    // Try "python3"
    if let Ok(x) = std::process::Command::new(bin).arg("python3").output() {
        for line in String::from_utf8(x.stdout).unwrap().lines() {
            if test_python_bin(line) {
                return Some(PathBuf::from(line));
            }
        }
    }

    // Try "python"
    if let Ok(x) = std::process::Command::new(bin).arg("python").output() {
        for line in String::from_utf8(x.stdout).unwrap().lines() {
            if test_python_bin(line) {
                return Some(PathBuf::from(line));
            }
        }
    }

    None
}

// ------------------------------------------------------------------------
// GIT CLONE & PREPARE TENSORFLOW SOURCE
// ------------------------------------------------------------------------

/// Clones TensorFlow from Git if not already cloned, preserving a shallow clone of the specified TAG.
fn prepare_tensorflow_source(tf_src_path: &Path) {
    let complete_clone_hint_file = tf_src_path.join(".complete_clone");
    if !complete_clone_hint_file.exists() {
        if tf_src_path.exists() {
            std::fs::remove_dir_all(tf_src_path).expect("Cannot remove existing tf_src_path");
        }
        let mut git = std::process::Command::new("git");
        git.arg("clone")
            .args(["--depth", "1"])
            .arg("--shallow-submodules")
            .args(["--branch", TAG])
            .arg("--single-branch")
            .arg(TF_GIT_URL)
            .arg(tf_src_path.as_os_str());

        println!("Cloning TensorFlow...");
        let start = Instant::now();
        if !git
            .status()
            .expect("Failed to execute `git clone`")
            .success()
        {
            panic!("git clone failed");
        }
        std::fs::File::create(&complete_clone_hint_file)
            .expect("Cannot create the .complete_clone marker");
        println!("Clone completed in {:?}", Instant::now() - start);
    }

    // If feature "xnnpack" is enabled, copy a special BUILD file
    #[cfg(feature = "xnnpack")]
    {
        let root = std::path::PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let bazel_build_path = root.join("build-res/tflitec_with_xnnpack_BUILD.bazel");
        let target = tf_src_path.join("tensorflow/lite/c/tmp/BUILD");
        std::fs::create_dir_all(target.parent().unwrap()).expect("Cannot create tmp directory");
        std::fs::copy(bazel_build_path, target)
            .expect("Cannot copy the temporary BUILD file for xnnpack");
    }
}

// ------------------------------------------------------------------------
// BAZEL BUILD LOGIC
// ------------------------------------------------------------------------

/// Configures TensorFlow (via `configure.py`) and then builds TFLiteC using Bazel.
fn build_tensorflow_with_bazel(tf_src_path: &str, config: &str, lib_output_path: &Path, os: &str) {
    // Determine the Bazel output path and Bazel target
    let (bazel_output_path_buf, bazel_target) = if os != "ios" {
        let ext = dll_extension();
        let sub_directory = if cfg!(feature = "xnnpack") {
            "/tmp"
        } else {
            ""
        };
        let mut lib_out_dir = PathBuf::from(tf_src_path)
            .join("bazel-bin")
            .join("tensorflow")
            .join("lite")
            .join("c");

        if !sub_directory.is_empty() {
            // If sub_directory == "/tmp", we join it
            lib_out_dir = lib_out_dir.join(&sub_directory[1..]);
        }
        let prefix = dll_prefix();
        let output_path = lib_out_dir.join(format!("{}tensorflowlite_c.{}", prefix, ext));
        let target_str = format!("//tensorflow/lite/c{}:tensorflowlite_c", sub_directory);
        (output_path, target_str)
    } else {
        // iOS uses a zip file (framework)
        let output_path = PathBuf::from(tf_src_path)
            .join("bazel-bin")
            .join("tensorflow")
            .join("lite")
            .join("ios")
            .join("TensorFlowLiteC_framework.zip");
        let target_str = String::from("//tensorflow/lite/ios:TensorFlowLiteC_framework");
        (output_path, target_str)
    };

    // 1) run `configure.py`
    let python_bin_path = env::var("PYTHON_BIN_PATH").expect("PYTHON_BIN_PATH is not set");
    if !std::process::Command::new(&python_bin_path)
        .arg("configure.py")
        .current_dir(tf_src_path)
        .status()
        .unwrap_or_else(|_| panic!("Cannot execute python at {}", &python_bin_path))
        .success()
    {
        panic!("TensorFlow configuration failed");
    }

    // 2) run `bazel build`
    let mut bazel = std::process::Command::new("bazel");
    {
        // Set bazel output_base under OUT_DIR to avoid conflicts on repeated builds
        let bazel_output_base_path = out_dir().join(format!("tensorflow_{}_output_base", TAG));
        bazel.arg(format!(
            "--output_base={}",
            bazel_output_base_path.to_string_lossy()
        ));
    }
    bazel.arg("build").arg("-c").arg("opt");

    // XNNPACK feature flags
    #[cfg(not(feature = "xnnpack"))]
    bazel.arg("--define").arg("tflite_with_xnnpack=false");
    #[cfg(any(feature = "xnnpack_qu8", feature = "xnnpack_qs8"))]
    bazel.arg("--define").arg("tflite_with_xnnpack=true");
    #[cfg(feature = "xnnpack_qs8")]
    bazel.arg("--define").arg("xnn_enable_qs8=true");
    #[cfg(feature = "xnnpack_qu8")]
    bazel.arg("--define").arg("xnn_enable_qu8=true");

    bazel
        .arg(format!("--config={}", config))
        .arg(&bazel_target)
        .current_dir(tf_src_path);

    // If user provided additional compiler options:
    if let Some(copts) = get_target_dependent_env_var(BAZEL_COPTS_ENV_VAR) {
        for opt in copts.split_ascii_whitespace() {
            bazel.arg(format!("--copt={opt}"));
        }
    }

    // iOS requires bitcode
    if os == "ios" {
        bazel.args(["--apple_bitcode=embedded", "--copt=-fembed-bitcode"]);
    }

    println!("Bazel Build Command: {:?}", bazel);
    if !bazel.status().expect("Unable to run bazel").success() {
        panic!("Failed to build TensorFlowLiteC via bazel");
    }
    if !bazel_output_path_buf.exists() {
        panic!(
            "The expected Bazel output was not found at {}",
            bazel_output_path_buf.display()
        );
    }

    // 3) Copy the resulting artifact(s) to OUT_DIR so Cargo can pick them up
    if os != "ios" {
        copy_or_overwrite(&bazel_output_path_buf, lib_output_path);

        // On Windows, also copy the `.lib` (import library) as `tensorflowlite_c.lib`
        if os == "windows" {
            let mut bazel_output_winlib_path_buf = bazel_output_path_buf.clone();
            bazel_output_winlib_path_buf.set_extension("dll.if.lib");
            let winlib_output_path_buf = out_dir().join("tensorflowlite_c.lib");
            copy_or_overwrite(&bazel_output_winlib_path_buf, &winlib_output_path_buf);
        }
    } else {
        // For iOS, unzip the framework zip
        if lib_output_path.exists() {
            std::fs::remove_dir_all(lib_output_path).unwrap();
        }
        let mut unzip = std::process::Command::new("unzip");
        unzip.args([
            "-q",
            bazel_output_path_buf.to_str().unwrap(),
            "-d",
            out_dir().to_str().unwrap(),
        ]);
        unzip.status().expect("Failed to execute unzip");
    }
}

// ------------------------------------------------------------------------
// PREBUILT INSTALLATION LOGIC
// ------------------------------------------------------------------------

/// Installs a prebuilt TFLiteC library into the OUT_DIR and copies/obtains the required headers.
fn install_prebuilt(prebuilt_tflitec_path: &str, tf_src_path: &Path, lib_output_path: &Path) {
    // 1) Copy the prebuilt library or framework
    copy_or_overwrite(PathBuf::from(prebuilt_tflitec_path), lib_output_path);

    // 2) On Windows, also copy the .lib
    if target_os() == "windows" {
        let mut from_path_lib = PathBuf::from(prebuilt_tflitec_path);
        from_path_lib.set_extension("lib");
        if !from_path_lib.exists() {
            panic!("A prebuilt .dll must have a matching .lib file in the same directory!");
        }
        let mut out_lib = lib_output_path.to_path_buf();
        out_lib.set_extension("lib");
        copy_or_overwrite(&from_path_lib, &out_lib);
    }

    // 3) Copy or download the required headers (c_api.h, c_api_types.h, etc.)
    let mut headers = vec![
        "tensorflow/lite/c/c_api.h",
        "tensorflow/lite/c/c_api_types.h",
    ];
    if cfg!(feature = "xnnpack") {
        headers.push("tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h");
        headers.push("tensorflow/lite/c/common.h");
    }
    copy_or_download_headers(tf_src_path, &headers);
}

// ------------------------------------------------------------------------
// HEADERS LOGIC (COPY / DOWNLOAD)
// ------------------------------------------------------------------------

/// Copies or downloads the specified header files into `tf_src_path`.
fn copy_or_download_headers(tf_src_path: &Path, file_paths: &[&str]) {
    if let Some(header_src_dir) = get_target_dependent_env_var(HEADER_DIR_ENV_VAR) {
        copy_headers(Path::new(&header_src_dir), tf_src_path, file_paths)
    } else {
        download_headers(tf_src_path, file_paths)
    }
}

/// Copies headers from `header_src_dir` to `tf_src_path`.
fn copy_headers(header_src_dir: &Path, tf_src_path: &Path, file_paths: &[&str]) {
    for file_path in file_paths {
        let dst_path = tf_src_path.join(file_path);
        if dst_path.exists() {
            continue;
        }
        if let Some(parent_dir) = dst_path.parent() {
            std::fs::create_dir_all(parent_dir).expect("Cannot create header directory");
        }
        copy_or_overwrite(header_src_dir.join(file_path), &dst_path);
    }
}

/// Downloads headers from GitHub if they are not found locally.
fn download_headers(tf_src_path: &Path, file_paths: &[&str]) {
    for file_path in file_paths {
        let download_path = tf_src_path.join(file_path);
        if download_path.exists() {
            continue;
        }
        if let Some(parent) = download_path.parent() {
            std::fs::create_dir_all(parent).expect("Cannot create header directory");
        }
        let url = format!(
            "https://raw.githubusercontent.com/tensorflow/tensorflow/{}/{}",
            TAG, file_path
        );
        download_file(&url, &download_path);
    }
}

/// Downloads a file from a URL using `curl` and writes to `path`.
fn download_file(url: &str, path: &Path) {
    let mut easy = curl::easy::Easy::new();
    let output_file = std::fs::File::create(path).unwrap();
    let mut writer = std::io::BufWriter::new(output_file);

    easy.url(url).unwrap();
    easy.write_function(move |data| Ok(writer.write(data).unwrap()))
        .unwrap();

    if let Err(e) = easy.perform() {
        // Remove the partially written file if download fails
        std::fs::remove_file(path).ok();
        panic!("Error occurred while downloading from {}: {:?}", url, e);
    }
}

// ------------------------------------------------------------------------
// FILE & DIRECTORY OPERATIONS
// ------------------------------------------------------------------------

/// Copies either a directory (recursively) or a single file from `src` to `dest`, overwriting if `dest` exists.
fn copy_or_overwrite<P: AsRef<Path> + Debug, Q: AsRef<Path> + Debug>(src: P, dest: Q) {
    let src_path: &Path = src.as_ref();
    let dest_path: &Path = dest.as_ref();

    // If the destination exists, remove it first
    if dest_path.exists() {
        if dest_path.is_file() {
            std::fs::remove_file(dest_path).expect("Cannot remove file");
        } else {
            std::fs::remove_dir_all(dest_path).expect("Cannot remove directory");
        }
    }

    // If source is a directory, copy recursively; if file, copy file
    if src_path.is_dir() {
        let options = fs_extra::dir::CopyOptions {
            copy_inside: true,
            ..fs_extra::dir::CopyOptions::new()
        };
        fs_extra::dir::copy(src_path, dest_path, &options).unwrap_or_else(|e| {
            panic!(
                "Cannot copy directory from {:?} to {:?}. Error: {}",
                src, dest, e
            )
        });
    } else {
        std::fs::copy(src_path, dest_path).unwrap_or_else(|e| {
            panic!(
                "Cannot copy file from {:?} to {:?}. Error: {}",
                src, dest, e
            )
        });
    }
}

// ------------------------------------------------------------------------
// DOCS.RS LOGIC
// ------------------------------------------------------------------------

/// If building on docs.rs (where the network is unavailable), extract pre-bundled resources (lib + bindings).
fn prepare_for_docsrs() {
    let library_path = out_dir().join("libtensorflowlite_c.so");
    let bindings_path = out_dir().join("bindings.rs");

    let mut unzip = std::process::Command::new("unzip");
    let root = std::path::PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    unzip
        .arg(root.join("build-res/docsrs_res.zip"))
        .arg("-d")
        .arg(out_dir());
    let success = unzip.status().map(|s| s.success()).unwrap_or(false);

    if !success || !library_path.exists() || !bindings_path.exists() {
        panic!("Failed to extract docs.rs resources");
    }
}

// ------------------------------------------------------------------------
// BINDINGS LOGIC
// ------------------------------------------------------------------------

/// Generates bindings via bindgen (for TFLite C API, and xnnpack delegate if enabled).
fn generate_bindings(tf_src_path: &Path) {
    let mut builder = bindgen::Builder::default().header(
        tf_src_path
            .join("tensorflow/lite/c/c_api.h")
            .to_string_lossy()
            .to_string(),
    );

    if cfg!(feature = "xnnpack") {
        builder = builder.header(
            tf_src_path
                .join("tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h")
                .to_string_lossy()
                .to_string(),
        );
    }

    let bindings = builder
        .clang_arg(format!("-I{}", tf_src_path.to_string_lossy()))
        // Re-generate if header changes
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_file = out_dir().join("bindings.rs");
    bindings
        .write_to_file(&out_file)
        .expect("Couldn't write bindings!");
}

// ------------------------------------------------------------------------
// LIB OUTPUT PATH / LINKAGE
// ------------------------------------------------------------------------

/// Returns the final path to the library or framework depending on the target OS.
fn lib_output_path(os: &str) -> PathBuf {
    if os != "ios" {
        let ext = dll_extension();
        let prefix = dll_prefix();
        out_dir().join(format!("{}tensorflowlite_c.{}", prefix, ext))
    } else {
        out_dir().join("TensorFlowLiteC.framework")
    }
}

/// For non-iOS, we specify a native search path and link the dylib; for iOS, we link a framework.
fn add_link_search_and_lib(os: &str, out_path: &Path) {
    if os != "ios" {
        println!("cargo:rustc-link-search=native={}", out_path.display());
        println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");
    } else {
        println!("cargo:rustc-link-search=framework={}", out_path.display());
        println!("cargo:rustc-link-lib=framework=TensorFlowLiteC");
    }
}
