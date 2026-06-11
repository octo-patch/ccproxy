{
  use = [
    { node = "@repo:mitmproxy"; mount = "inspector/mitmproxy"; }
    { node = "@repo:slirp4netns"; mount = "inspector/slirp4netns"; }
    { node = "@repo:xepor"; mount = "inspector/xepor"; }
    { node = "@repo:xepor-examples"; mount = "inspector/xepor-examples"; }
    { node = "@repo:jlowin-fastmcp"; mount = "lib/fastmcp"; }
    { node = "@repo:glom"; mount = "lib/glom"; }
    { node = "@repo:litellm"; mount = "lib/litellm"; }
    { node = "@repo:pydantic-ai"; mount = "lib/pydantic-ai"; }
    { node = "@repo:tyro"; mount = "lib/tyro"; }
  ];
  config = {
    auto_mount = true;
  };
}
