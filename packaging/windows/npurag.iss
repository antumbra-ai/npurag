; Inno Setup script for the npurag Windows installer.
;
; Built by .github/workflows/release.yml, which passes the version in:
;   iscc /DAppVersion=1.2.3 packaging\windows\npurag.iss
;
; The install is per-user and needs no administrator: npurag is a personal
; command-line tool that writes its index under the user's own profile, so
; asking for machine-wide privileges would buy nothing.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#define AppName "npurag"
#define AppPublisher "Grzegorz Oleksy"
#define AppURL "https://github.com/antumbra-ai/npurag"
#define AppExeName "npurag.exe"

[Setup]
; Never change AppId: it is how Windows recognises an upgrade of an existing
; installation rather than a second copy.
AppId={{6B1C9E2A-5F43-4B7C-9E21-0D5A7C3F8B14}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
VersionInfoVersion={#AppVersion}

DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; A command-line tool has nothing to show after installing.
DisableDirPage=auto
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

LicenseFile=..\..\LICENSE
OutputBaseFilename=npurag-{#AppVersion}-windows-x86_64-setup
OutputDir=..\..\dist
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The PATH entry below is an environment change; Windows needs telling so open
; shells are notified rather than silently keeping a stale PATH.
ChangesEnvironment=yes
UninstallDisplayName={#AppName} {#AppVersion}
UninstallDisplayIcon={app}\{#AppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "polish"; MessagesFile: "compiler:Languages\Polish.isl"

[Tasks]
Name: "addtopath"; Description: "Add npurag to my PATH (recommended)"; GroupDescription: "Command line:"

[Files]
Source: "..\..\dist\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\NOTICE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#AppName} on GitHub"; Filename: "{#AppURL}"

[Code]
const
  EnvironmentKey = 'Environment';

function PathSegments(const Path: string): TArrayOfString;
var
  Rest: string;
  Cut: Integer;
  Count: Integer;
begin
  SetArrayLength(Result, 0);
  Rest := Path;
  Count := 0;
  while Rest <> '' do
  begin
    Cut := Pos(';', Rest);
    if Cut = 0 then
    begin
      SetArrayLength(Result, Count + 1);
      Result[Count] := Trim(Rest);
      Rest := '';
    end
    else
    begin
      SetArrayLength(Result, Count + 1);
      Result[Count] := Trim(Copy(Rest, 1, Cut - 1));
      Rest := Copy(Rest, Cut + 1, Length(Rest) - Cut);
    end;
    Count := Count + 1;
  end;
end;

{ Compare paths case-insensitively and ignoring a trailing slash, so an entry
  added by an earlier install is recognised however it was written. }
function SamePath(const A, B: string): Boolean;
begin
  Result := CompareText(RemoveBackslashUnlessRoot(A), RemoveBackslashUnlessRoot(B)) = 0;
end;

function PathContains(const Path, Wanted: string): Boolean;
var
  Segments: TArrayOfString;
  I: Integer;
begin
  Result := False;
  Segments := PathSegments(Path);
  for I := 0 to GetArrayLength(Segments) - 1 do
    if SamePath(Segments[I], Wanted) then
    begin
      Result := True;
      Exit;
    end;
end;

procedure AddToPath(const Dir: string);
var
  Path: string;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, EnvironmentKey, 'Path', Path) then
    Path := '';
  if PathContains(Path, Dir) then
    Exit;
  if (Path <> '') and (Copy(Path, Length(Path), 1) <> ';') then
    Path := Path + ';';
  RegWriteExpandStringValue(HKEY_CURRENT_USER, EnvironmentKey, 'Path', Path + Dir);
end;

{ Remove only our own entry, leaving everything else exactly as it was. An
  uninstaller that rewrites the whole PATH is a good way to break a machine. }
procedure RemoveFromPath(const Dir: string);
var
  Path, Rebuilt: string;
  Segments: TArrayOfString;
  I: Integer;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, EnvironmentKey, 'Path', Path) then
    Exit;
  if not PathContains(Path, Dir) then
    Exit;

  Segments := PathSegments(Path);
  Rebuilt := '';
  for I := 0 to GetArrayLength(Segments) - 1 do
  begin
    if (Segments[I] = '') or SamePath(Segments[I], Dir) then
      Continue;
    if Rebuilt <> '' then
      Rebuilt := Rebuilt + ';';
    Rebuilt := Rebuilt + Segments[I];
  end;
  RegWriteExpandStringValue(HKEY_CURRENT_USER, EnvironmentKey, 'Path', Rebuilt);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('addtopath') then
    AddToPath(ExpandConstant('{app}'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveFromPath(ExpandConstant('{app}'));
end;
