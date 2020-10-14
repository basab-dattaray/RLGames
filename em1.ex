defmodule Em1 do
  @moduledoc false
  


  use Application

  def start(_type, _args) do
    Em1.Supervisor.start_link()
  end
end