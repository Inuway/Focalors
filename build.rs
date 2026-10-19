//! Windows only: embeds the app icon and the file-properties metadata
//! (product name, description, version) into focalors.exe, so the file
//! shows an icon in Explorer and carries the version in its properties.
//! On every other host this script does nothing. The resource compiler it
//! needs (rc.exe) ships with the MSVC toolchain that rustup's default
//! Windows target and GitHub's Windows runners use.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/icon/focalors.ico");

    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        // Version fields come from Cargo.toml automatically.
        res.set_icon("assets/icon/focalors.ico")
            .set("ProductName", "Focalors")
            .set("FileDescription", "Focalors Chess")
            .set("LegalCopyright", "GPL-3.0-or-later");
        res.compile()
            .expect("embedding the Windows icon and version info into focalors.exe");
    }
}
