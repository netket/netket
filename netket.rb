class Netket < Formula
  version="1.0.4"
  desc "Machine learning algorithms for many-body quantum systems "
  homepage "https://www.netket.org"
  url "https://github.com/netket/netket/archive/v"+version+".tar.gz"
  sha256 "0b344d526ee34d187281d0c2f7952c91728abbe22553e3dbcd45fcfeb312c3b5"
  depends_on "cmake" => :build
  depends_on "open-mpi"

  def install
    mkdir "build" do
      system "cmake", "-G", "Unix Makefiles", "..",*std_cmake_args,"-DBUILD_TESTING=OFF --config Release"
      system "make"
      system "make", "install"
    end
  end

  test do
    output=system `netket`
    assert (output.to_s.include? "1.0.3")
  end
end
