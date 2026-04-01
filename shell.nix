{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    nodejs_22
  ];

  shellHook = ''
    echo "Node.js $(node --version)"
    echo "npm $(npm --version)"
    echo ""
    echo "Run 'npm install' to install dependencies"
    echo "Run 'npm run docs:dev' to start the dev server"
  '';
}
